# EASFL revision: traffic playback firmware for FIT IoT-LAB M3 nodes (IEEE 802.15.4, AT86RF231)
APPLICATION = easfl_traffic
BOARD ?= iotlab-m3
RIOTBASE ?= $(CURDIR)/../RIOT

USEMODULE += netdev_default
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc
USEMODULE += gnrc_netif_hdr
USEMODULE += gnrc_pktbuf
USEMODULE += netstats_l2
USEMODULE += shell
USEMODULE += shell_cmds_default
USEMODULE += ztimer_usec
USEMODULE += ztimer_msec

DEVELHELP ?= 0
QUIET ?= 1
include $(RIOTBASE)/Makefile.include
