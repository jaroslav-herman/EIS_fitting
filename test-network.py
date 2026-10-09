import wepy.basics as we
folder = we.load_folders(r'\\ELECTROLYZER\PEM-WE_measurements\2026/', contains_string='468', mode='any')
print(folder)
#folder = r"\\ELECTROLYZER\PEM-WE_measurements\2026\468_VII_cathode_etching_series_210min"
print(folder)
files = we.load_files(folder[0], contains_string=['ay','rocedure','PEIS.mpr'], extension='.mpr', natural_sort=True)
print(files)