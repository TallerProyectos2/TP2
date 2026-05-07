import subprocess

class Net_artemis:
	def __init__(self):
		pass
	@staticmethod
	def get_SSID_actual():
		return subprocess.check_output(["iw","dev","mlan0","link"]).split('\n\t')[1][6:]
	@staticmethod
	def get_dbm_signal_actual():
		return int(subprocess.check_output(["iw","dev","mlan0","link"]).split('\n\t')[5][8:].rstrip(' dBm'))
	@staticmethod	
	def scan():
		return subprocess.check_output(["sudo","iw","dev","mlan0","scan"])
	@staticmethod
	def get_public_IP():
		try:
			public_IP=subprocess.check_output(["upnpc","-s"]).split("ExternalIPAddress = ")[1].split('\n')[0]
			type_conection="4G/5G"
			if public_IP == "10.0.103.56":
				public_IP=str(subprocess.check_output(["ip","addr"]).split(b'mlan')[1].split(b'inet')[1].split()[0]).split("'")[0]
				type_conection="Wifi"
		except:
			public_IP=str(subprocess.check_output(["ip","addr"]).split(b'mlan')[1].split(b'inet')[1].split()[0]).split("'")[0]
			type_conection="Wifi"
		return (type_conection,public_IP)