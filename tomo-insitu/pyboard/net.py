import network

n = network.WLAN(network.STA_IF)
n.active(True)
n.connect('dadou','lemotdepasse')
print("Connected!" if n.isconnected() else 'Failed to connect')
