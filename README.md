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
# Additional Information
The tool was created by Karol Saputa from the Institute of Computer Science, Polish Academy of Sciences.

This work was supported by the European Regional Development Fund as a part of 2014–2020 Smart Growth Operational Programme, CLARIN — Common Language Resources and Technology Infrastructure, project no. POIR.04.02.00-00C002/19 and by the project co-financed by the Minister of Education and Science under the agreement 2022/WK/09.
# Citation
Karol Saputa. 2022. Coreference Resolution for Polish: Improvements within the CRAC 2022 Shared Task. In Proceedings of the CRAC 2022 Shared Task on Multilingual Coreference Resolution, pages 18–22, Gyeongju, Republic of Korea. Association for Computational Linguistics.

```
@inproceedings{saputa-2022-coreference,
    title = "Coreference Resolution for {P}olish: Improvements within the {CRAC} 2022 Shared Task",
    author = "Saputa, Karol",
    editor = "{\v{Z}}abokrtsk{\'y}, Zden{\v{e}}k  and
      Ogrodniczuk, Maciej",
    booktitle = "Proceedings of the CRAC 2022 Shared Task on Multilingual Coreference Resolution",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.crac-mcr.2",
    pages = "18--22",
    abstract = "The paper presents our system for coreference resolution in Polish. We compare the system with previous works for the Polish language as well as with the multilingual approach in the CRAC 2022 Shared Task on Multilingual Coreference Resolution thanks to a universal, multilingual data format and evaluation tool. We discuss the accuracy, computational performance, and evaluation approach of the new System which is a faster, end-to-end solution.",
}
```

Copyright (c) 2022-2023 CLARIN-PL Wroclaw University of Technology and Science, Institute of Computer Science, Polish Academy of Sciences
