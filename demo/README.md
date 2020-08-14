## Simple Demo for Summarization Models
To help visualize the outputs of summarization models and understand what's going on under the hood.

### How to Run the demo
The demo is built with [Django](https://www.djangoproject.com/). 

First install all the required packages in `requirements.txt` 
```
$ pip install -r requirements.txt
```

Then go to `transformers/` directory in the root directory of this repo, install the transformers package from the source
 there (my own fork, with my custom generation function)
```
$ pip install .
``` 
To start the server
```
$ python manage.py runserver <ip-address>:<port-number>
```