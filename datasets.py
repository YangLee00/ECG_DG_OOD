from torch.utils.data import Dataset,DataLoader
import numpy as np
from imblearn.over_sampling import SMOTE

def onehot_2_one(innp):
    outnp=np.argmax(innp,axis=1)
    return outnp

def one_2_onehot(innp):
    outnp=np.zeros((innp.shape[0],np.max(innp)+1))
    for tmponehot in range(innp.shape[0]):
        outnp[tmponehot,innp[tmponehot]]=1
    return outnp
        
class GetLoader(Dataset):
    def __init__(self, dataname,train_val_test_split,mode='train',smote=False):
        data_list=[]
        label_list=[]
        self.total_class=[]
        
        dataset_num=len(dataname)
        
        if mode=='train':
            dataset_index=-1
            for i in dataname:
                dataset_index+=1
                print('train data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[:int(tmp_data.shape[0]*train_val_test_split[0]),:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[:int(tmp_label.shape[0]*train_val_test_split[0]),:]
                
                
                if smote==True:
                    
                    smo = SMOTE(n_jobs=-1) 
                    
                    tmp_label=onehot_2_one(tmp_label)
                    
                    print('-')
                    print(tmp_data.shape)
                    print(tmp_label.shape)
                    
                    tmp_new_data, tmp_new_label = smo.fit_resample(tmp_data[:,:,0], tmp_label) 
                    
                    tmp_data=np.expand_dims(tmp_new_data,axis=2)

                    tmp_label=one_2_onehot(tmp_new_label)
                    
                    print('smoted train data size : '+str(int(tmp_data.shape[0])))
                    print('smoted train label class : ',np.sum(tmp_label,axis=0))
                    
                tmp_label=np.pad(tmp_label, ((0,0),(0,dataset_num)))
                
                tmp_label[:,3+dataset_index]=1
                
                print('train data size : '+str(int(tmp_data.shape[0])))
                print('train label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total train data shape = ',self.data.shape)
            print('total train label shape = ',self.label.shape)
            print('total train data class = ',np.sum(self.label,axis=0))    
            
            self.total_class=np.sum(self.label,axis=0)[:3]
            
        elif mode=='val':
            for i in dataname:
                print('val data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[int(tmp_data.shape[0]*train_val_test_split[0]):int(tmp_data.shape[0]*(train_val_test_split[0]+train_val_test_split[1])),:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[int(tmp_label.shape[0]*train_val_test_split[0]):int(tmp_label.shape[0]*(train_val_test_split[0]+train_val_test_split[1])),:]
                print('val data size : '+str(int(tmp_data.shape[0])))
                print('val label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total val data shape = ',self.data.shape)
            print('total val label shape = ',self.label.shape)
            print('total val data class = ',np.sum(self.label,axis=0)) 
            
            self.total_class=np.sum(self.label,axis=0)[:3]
            
        elif mode=='test':
            for i in dataname:
                print('test data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[int(tmp_data.shape[0]*(1-train_val_test_split[2])):,:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[int(tmp_label.shape[0]*(1-train_val_test_split[2])):,:]
                print('test data size : '+str(int(tmp_data.shape[0])))
                print('test label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total test data shape = ',self.data.shape)
            print('total test label shape = ',self.label.shape)
            print('total test data class = ',np.sum(self.label,axis=0)) 
            
            self.total_class=np.sum(self.label,axis=0)[:3]
        else:
            print('dataloader mode error')
            
        
            
        
    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)


class train_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0,smote=False):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H']), 'Check your dataset name!'
        self.getloader=GetLoader(dataname,train_val_test_split,mode='train',smote=smote)
        self.bs=bs
        self.num_workers=num_workers

    @property
    def total_class(self):
        return self.getloader.total_class
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=True,drop_last=False,num_workers=self.num_workers)
    
class val_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H']), 'Check your dataset name!'
        self.getloader=GetLoader(dataname,train_val_test_split,mode='val')
        self.bs=bs
        self.num_workers=num_workers
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=False,drop_last=False,num_workers=self.num_workers)
    
class test_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H']), 'Check your dataset name!'
        self.getloader=GetLoader(dataname,train_val_test_split,mode='test')
        self.bs=bs
        self.num_workers=num_workers
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=False,drop_last=False,num_workers=self.num_workers)
    
    
    
    

    