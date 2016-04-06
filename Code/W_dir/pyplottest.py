import matplotlib.pyplot as plt
# Based on http://stackoverflow.com/a/8531491/190597 (Andrey Sobolev)

# fig = plt.figure()
# ax = fig.add_subplot(111)
y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]
# col_labels = ['col1', 'col2', 'col3']
# row_labels = ['row1', 'row2', 'row3']
# table_vals = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
table_vals = [[11, 12, 13, "rohliky", "kachna", "pes", "kocka", 11, 12, 13, "rohliky", "kachna", "pes", "kocka"],
			  [21, 22, 23, "housky", "kachna", "pes", "kocka", 11, 12, 13, "rohliky", "kachna", "pes", "kocka"],
			  [31, 32, 33, "utopence", "kachna", "pes", "kocka", 11, 12, 13, "rohliky", "kachna", "pes", "kocka"]]

print "number of columns: ", len(table_vals[0])

the_table = plt.table(cellText=table_vals,
                      colWidths=[0.08] * len(table_vals[0]),
                      # rowLabels=row_labels,
                      # colLabels=col_labels,
                      loc='center',
                      cellLoc="center")
plt.axis('off')
the_table.set_fontsize(50)
# the_table.scale(3, 4)

# plt.plot(y)
plt.show()