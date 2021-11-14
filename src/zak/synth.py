import threading
from time import time

import jack
import torch
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

zak = torch.jit.load('zak.pt').cuda()

amp = torch.zeros(1, 1, 1).cuda() * - 90.
freq = torch.zeros(1, 1, 1).cuda()

# Run once to build the computation graph
with torch.no_grad():
    _ = zak(freq, amp)

# Prepare jack client
client = jack.Client('zak-rt')

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()

buff = torch.zeros(448).cuda()
buff_len = 0


# TODO: Shuffling buffer from different threads is very slow in python.
# No escape from C/C++
@client.set_process_callback
def process(frames):
    global buff, buff_len
    assert len(client.inports) == len(client.outports)
    assert frames == client.blocksize
    for o in client.outports:
        now = time()
        with torch.no_grad():
            if buff_len == 0:
                out_signal = torch.cat((zak(freq, amp)[0, 0], zak(freq, amp)[0, 0]))
                out_signal, buff[:] = out_signal[:512], out_signal[512:]
                buff_len = 448
            else:
                out_signal = torch.cat((buff[-buff_len:], zak(freq, amp)[0, 0]))
                buff_len -= 32
                out_signal, buff[448 - buff_len:] = out_signal[:512], out_signal[512:]

            o.get_buffer()[:] = out_signal.cpu().squeeze(0).numpy()
        dur = time() - now
        if dur >= frames / 48000:
            print('missed a frame')


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)


# create two ports
client.inports.register('input_1')
client.outports.register('output_1')


def transport_handler(_, *args):
    global amp
    if args[0] in ['fastrewind', 'stop']:
        amp[...] = -80.


def vln_handler(_, *args):
    global amp, freq
    freq[...] = args[0]
    amp[...] = args[1]


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")


dispatcher = Dispatcher()
dispatcher.map("/transport", transport_handler)
dispatcher.map("/vln", vln_handler)
dispatcher.set_default_handler(default_handler)

ip = "127.0.0.1"
port = 1337

server = BlockingOSCUDPServer((ip, port), dispatcher)

with client:
    capture = client.get_ports(is_physical=True, is_output=True)
    if not capture:
        raise RuntimeError('No physical capture ports')

    for src, dest in zip(capture, client.inports):
        client.connect(src, dest)

    playback = client.get_ports(is_physical=True, is_input=True, is_audio=True)
    if not playback:
        raise RuntimeError('No physical playback ports')

    for src, dest in zip(client.outports, playback):
        client.connect(src, dest)

    print('Press Ctrl+C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
