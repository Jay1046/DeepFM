settings = {
    'sparse_features' : ['name', 'processor', 'ram', 'os', 'storage'],
    'dense_features' : ['price(in Rs.)','display(in inch)'],
    'input_columns' : ['name', 'processor', 'ram', 'os', 'storage', 'price(in Rs.)','display(in inch)'],
    'target_column' : ['rating'],
    'embedding_dim' : 4
}

path = 'C:/Users/user/Downloads/vscode/4월1주차DL/DeepFM/laptops.csv'