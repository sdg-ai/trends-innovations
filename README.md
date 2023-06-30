# Trends and Innovation Classifier
Identifies and locates mentions of particular climate change trends and innovations.

## Installation

### Pre-requisites
```
1. Python 3.*
```

### Installation
```shell
git clone https://github.com/sdg-ai/trends-innovations.git
pip3 install -r requirements.txt
python3 setup.py install
```

## Usage

```python
from experimental.trends_innovation_classifier import TrendsInnovationClassifier

classifier = TrendsInnovationClassifier()

text = < text >

output = classifier.predict(text)
```

### Example

```python
# Import libraries
import pandas as pd
from experimental.trends_innovation_classifier import TrendsInnovationClassifier

# Create classifier object
classifier = TrendsInnovationClassifier()

# Fetch the input text
with open('example_text.txt', 'rt', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Make predictions 
output = classifier.predict(text)

# Print the result
df = pd.DataFrame(output, columns=['string_indices', 'prediction', 'text'])
print(f"\n{df.to_string(index=False)}")
```

## License
sdg-ai/trends-innovations is licensed under the GNU General Public License v3.0 license, as found in the [LICENSE](LICENSE) file.
