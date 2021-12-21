from pgmpy.readwrite import BIFReader
from pgmpy.readwrite import XMLBIFWriter
import os
path = os.getcwd() + "/testing/"
path = path + 'mildew.bif'
with open(path) as f:
    bn_file = f.read()
bif_reader = BIFReader(string=bn_file)
model = model = bif_reader.get_model()
writer = XMLBIFWriter(model)
writer.write_xmlbif(f"mildew.BIFXML")