import pandas as pd
import numpy as np
def get_class_location(row):
    row["M"] = np.load(row["M"].replace('a','b'), "r")
    # "r": This string argument specifies that the file should be opened in read-only mode.

data = {
    'M': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 28],
    'City': ['New York', 'London', 'San Francisco']
}

# Create a DataFrame from the dictionary
train_df = pd.DataFrame(data)

train_df = train_df.apply(get_class_location, axis = 1)



# axis=1: This parameter indicates that the function 
# should be applied along the rows (axis 1) of the DataFrame.

# npy and npz are file formats used by the NumPy library in Python for storing and loading arrays and data structures efficiently. These formats are designed to maintain the array data, its shape, and other associated metadata. They are commonly used for saving and sharing large arrays and data structures, especially in scientific computing and data analysis.

# .npy Format:
# The .npy format is used to store a single NumPy array in a binary file. This format preserves the array's data type, shape, and other metadata. .npy files are compact and efficient for storing large arrays. You can use the np.save() function to save an array to a .npy file and the np.load() function to load it back.

# Example usage:
# import numpy as np

# data = np.array([1, 2, 3])
# np.save('data.npy', data)  # Save the array
# loaded_data = np.load('data.npy')  # Load the array

# .npz Format:
# The .npz format is used to store multiple NumPy arrays (or other data) in a single compressed archive file. Each array is saved under a unique name within the .npz file. This is useful when you want to save and load multiple arrays together. You can use the np.savez() or np.savez_compressed() function to create .npz files and the np.load() function to load them.

# Example usage:
# Copy code
# import numpy as np

# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])
# np.savez('arrays.npz', a=array1, b=array2)  # Save multiple arrays
# loaded_data = np.load('arrays.npz')  # Load the arrays

# array1_loaded = loaded_data['a']
# array2_loaded = loaded_data['b']
# Both .npy and .npz files provide a convenient way to save and load arrays without losing data integrity. They are widely used in scientific computing, machine learning, and data analysis applications.