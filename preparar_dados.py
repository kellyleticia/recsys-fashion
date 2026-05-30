import pandas as pd
import json

styles = pd.read_csv('data/styles.csv', on_bad_lines='skip')
images = pd.read_csv('data/images.csv', on_bad_lines='skip')

styles = styles.dropna(subset=['productDisplayName'])

for col in ['baseColour', 'season', 'usage', 'articleType', 'subCategory', 'gender']:
    styles[col] = styles[col].fillna('')

images['id'] = images['filename'].str.replace('.jpg', '', regex=False).astype(int)

df = styles.merge(images[['id', 'link']], on='id', how='left')

sem_imagem = df['link'].isna().sum()
print(f"Produtos sem imagem: {sem_imagem}")
print(f"Produtos com imagem: {df['link'].notna().sum()}")

df['texto_embedding'] = (
    df['productDisplayName'] + ' ' +
    df['articleType'] + ' ' +
    df['subCategory'] + ' ' +
    df['gender'] + ' ' +
    df['baseColour'] + ' ' +
    df['season'] + ' ' +
    df['usage']
).str.strip()

produtos = df[[
    'id', 'productDisplayName', 'gender',
    'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'usage', 'link', 'texto_embedding'
]].to_dict(orient='records')

with open('data/produtos.json', 'w', encoding='utf-8') as f:
    json.dump(produtos, f, ensure_ascii=False, indent=2)

print(f"\n {len(produtos)} produtos salvos em data/produtos.json")
print("\nExemplo:")
print(json.dumps(produtos[0], ensure_ascii=False, indent=2))