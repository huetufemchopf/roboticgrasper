import urdf_parser_py
import os
import xml.etree.ElementTree as ET

data = 'random_urdfs'

def accesssizes(datapath):
    newsize = "0.01 0.01 0.01"
    for folder in os.listdir(datapath):
        if 'DS_Store' not in folder:

            for file in os.listdir(os.path.join(datapath, folder)):
                if '.urdf' in file:
                    tree = ET.parse(os.path.join(datapath, folder,file))
                    root = tree.getroot()
                    for mesh in root.iter('mesh'):
                        mesh.set('scale', "0.01 0.01 0.01")
                        print(mesh.attrib)

                    tree.write(os.path.join(datapath, folder, file))
    print('finished')

accesssizes(data)