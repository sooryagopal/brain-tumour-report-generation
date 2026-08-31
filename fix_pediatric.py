import os
from extract_2d_slices import process_brats_folder

src_dir = "/Users/macbookpro/Downloads/DATA/Pediotric glioma"
dest_train = "/Users/macbookpro/Downloads/FInal Year Project/Project/data/Training"
dest_test = "/Users/macbookpro/Downloads/FInal Year Project/Project/data/Testing"

process_brats_folder(os.path.join(src_dir, "Training"), dest_train, dest_test, "pediatric_glioma")
process_brats_folder(os.path.join(src_dir, "Validation"), dest_train, dest_test, "pediatric_glioma")
print("Finished fixing pediatric_glioma")
