import xml.etree.ElementTree as etree

tree = etree.parse('training-data.data.xml')
root = tree.getroot()


for lexelt in tree.findall('lexelt'):
  for i in lexelt.findall('instance'):
    print i.get('id')