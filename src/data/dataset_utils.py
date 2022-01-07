import json
import os

def new_annotation_structure(anno_dict,dataset):
    annotations = anno_dict['annotations']
    images = anno_dict['images']
    annotations_new = []
    i = 0
    while i < len(annotations):
        image_id_prev = annotations[i]['image_id']
        image_path=os.path.join('data',dataset,images[image_id_prev]['file_name'])
        dict_t = {'image_id': image_id_prev, 'image_location':image_path,'annotations': []}
        while image_id_prev == annotations[i]['image_id']:
            dict_t['annotations'].append(annotations[i])
            i += 1
            if i >= len(annotations):
                break
        annotations_new.append(dict_t)
    return annotations_new


def generate_new_annotation_file(dataset=None):
    path = os.path.join(os.getcwd(), 'data', dataset)
    file = open(os.path.join(path, '_annotations.coco.json'), 'r')
    d = json.load(file)
    file.close()
    annotations = new_annotation_structure(d,dataset)
    annotation_path=os.path.join(path,'new_annotations.json')
    if os.path.exists(annotation_path):
        os.remove(annotation_path)
    annotation_file = open(annotation_path,'w')
    json.dump(annotations, annotation_file)
