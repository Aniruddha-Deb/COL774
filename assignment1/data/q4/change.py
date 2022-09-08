import csv

with open('q4x.dat') as dat_file, open('X.csv', 'w') as csv_file:
    data = []
    for line in dat_file:
        row = [field.strip() for field in line.split(' ')]
        csv_file.write(row[0] +"," + row[2] + "\n")
    
with open('q4y.dat') as dat_file, open('Y.csv', 'w') as csv_file:
    data = []
    for line in dat_file:
        row = [field.strip() for field in line.split(' ')]
        csv_file.write(row[0]+"\n")
        