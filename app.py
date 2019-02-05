import json
from flask import Flask, abort,request, render_template,jsonify
from flask_restful import reqparse, abort, Api, Resource
import numpy
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)

api = Api(app)

class Base(Resource):
    def get(self):
        return {'message': 'I am working fine'}

class Predict(Resource):
 
    def post(self):
        print("got here")
        lr = joblib.load('model.pkl')

        # Saving the data columns from training
        if 'file' in request.files: 
            f = request.files['file']
        #model_columns = ['customer_value', 'customer_class', 'spend_total', 'campaign_type']
   

        user_input = [{f}]
        query_df = pd.DataFrame(user_input)
        query = pd.get_dummies(query_df)
        prediction = lr.predict(query)
        print(list(prediction))
        return jsonify({'prediction': list(prediction)})
    
    def get(self):
        return self.post()

#class MyEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, numpy.integer):
#            return int(obj)
#        elif isinstance(obj, numpy.floating):
#           return float(obj)
#        elif isinstance(obj, numpy.ndarray):
#            return obj.tolist()
#        else:
#            return super(MyEncoder, self).default(obj)



#def get_delay():
    #clf_path = 'models.pkl'
    #with open(clf_path, 'rb') as f:
    #    u = pickle._Unpickler(f)
    #    u.encoding = 'latin1'
    #    lr = u.load()


    #total_spent = result['spend_total']
    #custom_class = result['customer_class']
    #customer_value = result['customer_value']
    #campaign_type = result['campaign_type']
    #user_input = [{'total_spent':total_spent}, {'custom_class':custom_class}, {'customer_value':customer_value}, {'campaign_type':campaign_type}]
    #query = pd.get_dummies(pd.DataFrame(user_input))
    #query = query.reindex(columns=model_columns, fill_value=0)
    #prediction = list(lr.predict(query))
    #return json.dumps({'prediction':prediction})
    #return json.dumps({'prediction': numpy.int32(prediction[0])}, cls=MyEncoder)
    #return jsonify({'prediction': str(prediction)})
    #return jsonify({'prediction': numpy.int32(prediction[0])}, cls=MyEncoder)

api.add_resource(Predict, '/api')    
api.add_resource(Base, '/')

if __name__ == '__main__':
    app.run(debug=True)


   