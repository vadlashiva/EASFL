/*
 * EASFL revision (IoT-76474-2026): traffic playback for energy measurement on FIT IoT-LAB M3.
 *
 * The node keeps its radio in receive mode (no duty cycling) and, on shell command, transmits a
 * payload of <nbytes> fragmented into IEEE 802.15.4 frames of <frame> payload bytes, unicast with
 * MAC acknowledgements and CSMA/CA retries (radio defaults), or broadcast. The controller script on
 * the IoT-LAB frontend replays one federated round of each scheme through these commands while the
 * testbed records the node's power, so contention, retransmissions and idle listening are real.
 *
 *   burst <l2addr|bcast> <nbytes> [frame=100] [gap_ms=2]   send one payload
 *   stats                                                   print frame counters (netstats_l2)
 *   mark <text>                                             print a timestamped marker
 *   ifconfig                                                (default) shows the short address
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/netstats.h"
#include "net/netif.h"
#include "shell.h"
#include "ztimer.h"

static uint8_t buf[127];
static uint32_t n_sent, n_err;

static gnrc_netif_t *radio(void)
{
    return gnrc_netif_iter(NULL);
}

static int send_frame(gnrc_netif_t *netif, const uint8_t *addr, size_t addr_len, int bcast, size_t len)
{
    gnrc_pktsnip_t *pkt = gnrc_pktbuf_add(NULL, buf, len, GNRC_NETTYPE_UNDEF);
    if (!pkt) {
        return -1;
    }
    gnrc_pktsnip_t *hdr = gnrc_netif_hdr_build(NULL, 0, addr, addr_len);
    if (!hdr) {
        gnrc_pktbuf_release(pkt);
        return -1;
    }
    pkt = gnrc_pkt_prepend(pkt, hdr);
    ((gnrc_netif_hdr_t *)hdr->data)->flags = bcast ? GNRC_NETIF_HDR_FLAGS_BROADCAST : 0;
    if (gnrc_netif_send(netif, pkt) < 1) {
        gnrc_pktbuf_release(pkt);
        return -1;
    }
    return 0;
}

static int cmd_burst(int argc, char **argv)
{
    if (argc < 3) {
        puts("usage: burst <l2addr|bcast> <nbytes> [frame=100] [gap_ms=2]");
        return 1;
    }
    gnrc_netif_t *netif = radio();
    uint8_t addr[GNRC_NETIF_L2ADDR_MAXLEN];
    size_t addr_len = 0;
    int bcast = (strcmp(argv[1], "bcast") == 0);
    if (!bcast) {
        addr_len = gnrc_netif_addr_from_str(argv[1], addr);
        if (addr_len == 0) {
            puts("error: bad address");
            return 1;
        }
    }
    long nbytes = atol(argv[2]);
    size_t frame = (argc > 3) ? (size_t)atoi(argv[3]) : 100;
    uint32_t gap = (argc > 4) ? (uint32_t)atoi(argv[4]) : 2;
    if (frame < 1 || frame > 100) {
        frame = 100;
    }
    uint32_t t0 = ztimer_now(ZTIMER_USEC);
    unsigned nf = 0, ne = 0;
    while (nbytes > 0) {
        size_t len = (nbytes > (long)frame) ? frame : (size_t)nbytes;
        memset(buf, 0xA5 ^ (uint8_t)nf, len);
        if (send_frame(netif, addr, addr_len, bcast, len) < 0) {
            ne++;
        }
        nf++;
        nbytes -= (long)len;
        ztimer_sleep(ZTIMER_MSEC, gap);    /* let the MAC finish (ACK wait, CSMA backoff) */
    }
    n_sent += nf; n_err += ne;
    printf("BURST dst=%s frames=%u errors=%u t_us=%lu\n", argv[1], nf, ne,
           (unsigned long)(ztimer_now(ZTIMER_USEC) - t0));
    return 0;
}

static int cmd_stats(int argc, char **argv)
{
    (void)argc; (void)argv;
    netstats_t *s = NULL;
    if (netif_get_opt(&radio()->netif, NETOPT_STATS, NETSTATS_LAYER2, &s, sizeof(s)) < 0 || !s) {
        puts("error: no stats");
        return 1;
    }
    printf("STATS tx_unicast=%lu tx_mcast=%lu tx_success=%lu tx_failed=%lu rx=%lu rx_bytes=%lu tx_bytes=%lu app_sent=%lu app_err=%lu\n",
           (unsigned long)s->tx_unicast_count, (unsigned long)s->tx_mcast_count,
           (unsigned long)s->tx_success, (unsigned long)s->tx_failed,
           (unsigned long)s->rx_count, (unsigned long)s->rx_bytes, (unsigned long)s->tx_bytes,
           (unsigned long)n_sent, (unsigned long)n_err);
    return 0;
}

static int cmd_mark(int argc, char **argv)
{
    printf("MARK t_us=%lu %s\n", (unsigned long)ztimer_now(ZTIMER_USEC), argc > 1 ? argv[1] : "");
    return 0;
}

static const shell_command_t cmds[] = {
    { "burst", "send a payload as 802.15.4 frames", cmd_burst },
    { "stats", "print link-layer frame counters", cmd_stats },
    { "mark", "print a timestamped marker", cmd_mark },
    { NULL, NULL, NULL }
};

int main(void)
{
    puts("EASFL traffic playback ready");
    char line_buf[SHELL_DEFAULT_BUFSIZE];
    shell_run(cmds, line_buf, SHELL_DEFAULT_BUFSIZE);
    return 0;
}
