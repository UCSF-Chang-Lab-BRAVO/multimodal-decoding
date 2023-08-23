def clean_labels(labels):
    """
    in : pandas dataframe of the labels
    out: 
    
    labels - the same dataframe, but the phone labels will have
        stress markings removed, and all commas will be removed. 
    all_ph - list of all the phonemes to make the encdoing dict.
    """

    newlabs= []
    all_ph = []
    for p in labels['ph_label']:
        pp_ = []
        for pp in p:
            if not ',' in pp:
                pp_.append(pp)
        p = pp_
        newlabs.append([pp[:2] for pp in p if not p == ','])
        all_ph.extend([pp[:2] for pp in p if not p == ','])
    labels['ph_label'] = newlabs
    return labels, all_ph