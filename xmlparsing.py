# from xml.dom import minidom
# xmldoc = minidom.parse('/home/srikar/Documents/ConcoDiscoDataset/Europarl-ConcoDisco-master/xml-files/ep-00-01-17.xml')
# #itemlist = xmldoc.getElementsByTagName('DiscourseConnective')

# for node in xmldoc.getElementsByTagName('ParallelChunk'):
# 	print node.firstChild.data

# #print(itemlist[0].attributes['name'].value)
# #for s in itemlist:
# #    print(s.attributes['name'].value)

import xml.etree.ElementTree as ET
tree = ET.parse('/home/srikar/Documents/ConcoDiscoDataset/Europarl-ConcoDisco-master/xml-files/ep-00-01-17.xml')
root = tree.getroot()
# for child in root.iter('en'):
# 	child2 = child
# 	for child1 in child.iter('DiscourseConnective'):
# 		print child1.text+"<-Connective, sentence->"+child2.text+"\n"
# 	temp = child

# for child in root.findall('en'):
# 	print child.find('DiscourseConnective').text

# for child in root.findall("./Speaker/ParallelChunk/en [Alignment]"):
# 	print child.text
# 	print "----------------------------"
sentence = ''
for x in root.iter('en'):
	flag = False

	for y in x:
		for z in y:
			if z.tag == 'DiscourseConnective':
				flag = True
		if y.tag == 'DiscourseConnective':
			flag = True

	if flag:
		if not x.text.isspace():
			sentence = x.text
		for y in x:
			for z in y:
				if not z.text.isspace():
					sentence = sentence+z.text
			if not y.text.isspace():
				sentence = sentence+y.text 
			sentence = sentence+y.tail
		print sentence