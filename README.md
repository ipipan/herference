# herference

## API Docs


### Installation

```
python3 -m pip install .
```

### Loading model

```
import herference
cr = herference.Herference()

```

### Inference

```
text = cr.predict('Twój przykładowy tekst, którego analizę chcesz przeprowadzić.')
```

### Access

```
print(text)
```

```

for cluster in text.clusters:
  for mention in cluster:
      print(f"{mention.text} {mention.subtoken_indices}")
      
print(text.tokenized)
```
