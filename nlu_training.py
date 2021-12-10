from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config


# loading the nlu training samples
training_data = load_data("data/nlu.md")


# trainer to educate our pipeline
trainer = Trainer(config.load("config.yaml"))

# train the model!
interpreter = trainer.train(training_data)

# store it for future use
model_directory = trainer.persist("models/nlu", fixed_model_name="current")




# A helper function for prettier output
import json

def pprint(o):   
    print(json.dumps(o, indent=2))

#pprint(interpreter.parse("mumbai"))