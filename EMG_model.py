from sklearn.externals import joblib

class EMG_Model():

    def prepare_data(self,intended_movement_labels,rows=8000):

        rep = []
        reps =rows // 6 if rows % 6 == 0 else (rows //6)+1

        for i in range(1,7):
            for j in range(0,reps):
                rep.append(i)

        rep = rep[:rows]
        emg_set = {}

        for i in intended_movement_labels:
            emg_set[i] = pd.read_csv('models/' +str(i)+".csv" ,nrows =rows,header=None)
            emg_set[i]['label'] = i
            emg_set[i].columns = [1,2,3,4,5,6,7,8,'label']
            emg_set[i]['rep'] = rep

        data = pd.DataFrame()

        for i in intended_movement_labels:
            data = pd.concat([data,emg_set[i]])

        data = data.drop_duplicates().reset_index(drop=True)
        dataLabel=data['label']
        dataRep=data['rep']
        data=data.drop(['label','rep'],1)

        normalized_emg=filteration (data,sample_rate=200)

        normalized_emg['label'] = dataLabel

        normalized_emg['rep'] = dataRep
        normalized_emg=normalized_emg.set_index('rep')
        rep_train=[1,3,6,4]
        normalized_emg_train,LL_train=prepare_df(rep_train,normalized_emg)
        predictors_train,outcomes_train=get_predictors_and_outcomes(intended_movement_labels,rep_train,normalized_emg_train,LL_train)

        #prepare test part
        rep_test=[2,5]
        normalized_emg_test,LL_test=prepare_df(rep_test,normalized_emg)

        #normalized_emg_test
        predictors_test,outcomes_test=get_predictors_and_outcomes(intended_movement_labels,rep_test,normalized_emg_test,LL_test)

        predictors_test = get_predictors(normalized_emg_test)
        return predictors_train,outcomes_train,predictors_test,outcomes_test

    def svm_model(self,predictors_train,outcomes_train):

        model=svm.LinearSVC(dual=False) # at C= 0.05:0.09 gives little increase in accuracy, around 0.4%
        model.fit(predictors_train,outcomes_train)
        return model

    def accuracy(self,model):
        return model.score(predictors_test,outcomes_test)*100

    def save_model(model,filename):
        joblib.dump(model, filename)



    def all_steps(self,movements,file_name):
        from sklearn.externals import joblib

        predictors_train,outcomes_train,predictors_test,outcomes_test = self.prepare_data(movements)
        model = self.svm_model(predictors_train,outcomes_train)

        #if you wanna accuracy
        print (self.accuracy(model))

        #save pickle
        self.save_model(model,file_name)

