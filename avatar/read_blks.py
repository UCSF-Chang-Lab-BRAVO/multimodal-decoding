from RT import *
import numpy as np
import pandas as pd
def read_a_block(subject='test_br_3',result_num=14):
    

    with savedData.DataInterface(result_num=result_num, subject=subject) as di:
    # get neural data 

        try:
            hg_z = di['/neural/hgacar200_running30'].values
            lfs_z = di['/neural/rawcar200_running30'].values
            # concatenate hg and lfs 
            neural = np.concatenate((hg_z,lfs_z),axis=1)
        except Exception:
            try: 
                print('using hgarawcar200_running30')
                neural = di['neural/hgarawcar200_running30'].values
            except Exception:
                print('couldnt load data, rip')
        # get the identity of each phrase in the set 
#         import ipdb
#         ipdb.set_trace()
        txt_lab = di.extract_event_targets(convert_labels=True)
        ind_lab = di.extract_event_targets()
        print(ind_lab, txt_lab)
        #     word_level = di.extract_component_targets()

        #     is_span =[]
        #     for p in txt_lab[0]: is_span.append("spanish" in p)
        #     # get the event table 
        #     is_span_word = []
        #     for ind,word in word_level.iterrows(): is_span_word.append(is_span[word['event_num']])
        #     word_level['is_span'] = is_span_word


        evt_tb = di.events
        timestamp = di['/general/timestamp']
    return neural,txt_lab,ind_lab,evt_tb,timestamp


def read_in_blocks(subject='test_br_3',blocks=[14],cue_alignment=[3,4],make_one_hot=False):
    ### INPUTS:
    
    # blocks -- list of blocks to pull
    # cue_alignment -- [seconds before cue, seconds after cue] to align each trial
    # make_one_hot -- whether or not to one hot encode the labels (bool)
    
    
    ### RETURNS:
    
    # blk_dat -- dataframe where each row is a block and the columns are 
    #                   blk: blk num, txt_lab: text ids of utterances
    #                   timestamp: date, start_times: times of go cue
    #                   aligned neural: neural data aligned to cue with cue_alignment times (hg then raw for 253 channels)
    #                   neural_z: neural data from block unaligned (hg then raw for 253 channels)
    #                   ind_lab: the index in task gimlet labels of utterances
    
    
    # unique_txt -- np.array of the text that was one hot encoded (first element maps to 0, second to 1 ....)
    
    # Define the lists that will end up comprising the block dictionary 
    print(blocks)
    all_neural = []
    all_txt_lab = []
    all_ind_lab = []
    all_start_times = []
    all_timestamp = []
    all_blks = []
    all_aligned = []    
    # iterate over blocks
    for ind,blk in enumerate(blocks):
        print(blk)
        try: 
            neural_z,txt_lab,ind_lab,evt_tb,timestamp = read_a_block(subject=subject,result_num=blk)
        except Exception: 
            print('Block failed', blk)
            continue
        all_txt_lab.append(txt_lab[0])
        all_start_times.append(txt_lab[1])
        all_ind_lab.append(ind_lab[0])
        all_timestamp.append(timestamp)
        all_neural.append(neural_z)
        all_blks.append(blk)
        # [1,2.5]
        all_aligned.append(align(neural_z,txt_lab[1],cue_alignment))
    # make the df 
    blk_dict = {}
    blk_dict['blk'] = all_blks
    blk_dict['txt_lab'] = all_txt_lab
    #blk_dict['neural'] = all_neural
    blk_dict['timestamp'] = all_timestamp
    blk_dict['start_times'] = all_start_times
    blk_dict['aligned_neural'] = all_aligned
    blk_dict['neural_z'] = all_neural
    blk_dict['ind_lab'] = all_ind_lab
    # one hot encode
    if(make_one_hot):
        unroll_txt_lab = []
        for blk_txt_lab in all_txt_lab:
            for txt_lab in blk_txt_lab:
                unroll_txt_lab.append(txt_lab)
        unique_txt = np.unique(unroll_txt_lab)
        print('label order', unique_txt)
        one_hot_labs = []
        for cur_txt_lab in all_txt_lab:
            cur_one_hot = []
            for lab in cur_txt_lab:
                cur_one_hot.append(np.where(unique_txt == lab)[0][0])
            one_hot_labs.append(cur_one_hot)
        blk_dict['one_hot_lab'] = one_hot_labs
                
        
    blk_dat = pd.DataFrame(blk_dict)
    
    if(make_one_hot):
        return(blk_dat,unique_txt)
    else:
        return(blk_dat)

def align(dat,times,pads,fs=200):
    # INPUTS:
        # dat: T x N matrix
        # times: list of timepoints (not samples)
        # pads: [t_before,t_after]
        # fs: sampling frequency   
    samples = (times*fs).astype(int)
    aligned = np.zeros((times.shape[0],int(sum(pads*fs)),dat.shape[1]))
    for ind,start in enumerate(samples):
        try:
            aligned[ind,:,:] = dat[start-int(pads[0]*fs):start+int(pads[1]*fs),:]
        except Exception: 
            data = dat[start-int(pads[0]*fs):start+int(pads[1]*fs),:]
            aligned[ind,:len(data), :] = data
            print('end of trial except')

    return(aligned)

