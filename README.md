# herference

Coreference Resolution for Polish language

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
----------------------------------------------
Copyright (c) 2022-2023 CLARIN-PL Wroclaw University of Technology and Science, Institute of Computer Science, Polish Academy of Sciences
