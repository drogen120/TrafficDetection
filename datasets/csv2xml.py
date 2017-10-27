import csv

csvFile = '/home/gpu_server2/DataSet/dayTrain/dayClip13/frameAnnotationsBOX.csv'
xmlFile = '/home/gpu_server2/DataSet/dayTrain/dayClip13/frameAnnotationsBOX.xml'

csvData = csv.reader(open(csvFile), delimiter = ';')
xmlData = open(xmlFile, 'w')
xmlData.write('<?xml version="1.0"?>' + "\n")
# there must be only one top-level tag
xmlData.write('<csv_data>' + "\n")

rowNum = 0
before_file = ''
for row in csvData:

    if rowNum == 0:
        tags = row
        # replace spaces w/ underscores in tag names
        for i in range(len(tags)):
            tags[i] = tags[i].replace(' ', '_')
    else:
        image_name = row[0].split('/')[1]
        if before_file == image_name:
            xmlData.write('<object>' + "\n")
            for i in range(1,len(tags)):
                xmlData.write('    ' + '<' + tags[i] + '>' \
                              + row[i] + '</' + tags[i] + '>' + "\n")
            xmlData.write('</object>' + "\n")
        else:
            if before_file != "":
                xmlData.write('</%s>' % before_file + "\n")
            xmlData.write('<%s>' % image_name + "\n")
            xmlData.write('<object>' + "\n")
            for i in range(1,len(tags)):
                xmlData.write('    ' + '<' + tags[i] + '>' \
                              + row[i] + '</' + tags[i] + '>' + "\n")
            xmlData.write('</object>' + "\n")
        before_file = image_name

    rowNum +=1

xmlData.write('</%s>' % before_file + "\n")
xmlData.write('</csv_data>' + "\n")
xmlData.close()
