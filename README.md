# evalServer


## On local machine (server to host the model)
Step 1 : install libraries

`pip install fastapi uvicorn transformers torch ngrok`

Step 2: replace the classification model on `server.py` and run it (preferably with GPU)

Step 3: create an [ngrok account](https://dashboard.ngrok.com/login) and obtain the auth-key from their website

Step 4: Run the following command
```
ngrok config add-authtoken YOUR_AUTHTOKEN
ngrok http 8000
```

Copy the output url `https://xxxx-xxxx.ngrok-free.app` and use it to replace API_URL in the `client_eval.py` script 
Note that for the free version, the url will change everytime you run `ngrok http 8000`.

## On the client machine (e.g., Codabench)

Step 5: Update the `client_eval.py` to match the desired output.


That's it
