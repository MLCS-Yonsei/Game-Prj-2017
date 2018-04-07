from os import listdir
from os.path import isfile, join
import cv2
import sys
import json
from random import shuffle

_valid_ratio = 0.2
_test_ratio = 0.2

_temp_files = []
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# Load original images
original_dir = './original'
augmented_dir = './augmented'
imgs = [f for f in listdir(original_dir) if isfile(join(original_dir, f))]

# results
result = {}
result['ids'] = []
result['image_info'] = {}

result['training_ids'] = {
    "age": [],
    "gender": []
}
result['validation_ids'] = {
    "age": [],
    "gender": []
}
result['test_ids'] = {
    "age": [],
    "gender": []
}

def image_process(img, l):
    _id = img.split('_')[0]

    # Copy Original Image
    _img = cv2.imread(original_dir + '/' + img)
    cv2.imwrite(augmented_dir + '/' + _id + '.jpg' ,_img)

    # Save Horizontally flipped Image
    _horizontal_img = cv2.flip( _img, 1)
    cv2.imwrite(augmented_dir + '/' + _id + '_f.jpg' ,_horizontal_img)
    l.append(_id + '_f')

    # Rotates both images
    _h, _w, _c = _img.shape

    _d = int(((_w*_w + _h*_h)**0.5)) 
    _rc = int(_w/2), int(_h/2)

    for a in [-5, 5]:
        _angle = a    
        _scale = 1

        _r_image = cv2.getRotationMatrix2D(_rc, _angle, _scale)
        _rotate_img = cv2.warpAffine(_img, _r_image, (_h, _w))
        cv2.imwrite(augmented_dir + '/' + _id + '_' + str(a) + '.jpg' ,_rotate_img)
        l.append(_id + '_' + str(a))

        _r_image_f = cv2.getRotationMatrix2D(_rc, _angle, _scale)
        _rotate_img_f = cv2.warpAffine(_horizontal_img, _r_image, (_h, _w))
        cv2.imwrite(augmented_dir + '/' + _id + '_f_' + str(a) + '.jpg' ,_rotate_img_f)
        l.append(_id + '_f_' + str(a))

for i, img in enumerate(imgs):
    printProgress(i + 1, len(imgs), 'Loading Images:', 'Done!', 2, 50)

    _id = img.split('_')[0]
    _age = img.split('_')[1]
    if img.split('_')[2] == 'male.jpg':
        _gender = 'M'
    elif img.split('_')[2] == 'female.jpg':
        _gender = 'F'
    else: 
        _gender = 'NULL'

    if _gender != 'NULL':
        result['ids'].append(_id)
        
        _r = {"age": _age, "gender": _gender}
        result['image_info'][_id] = _r

labels = {
    "M": [],
    "F": [],
    "0": [],
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": []
}
for i, id in enumerate(imgs):
    printProgress(i + 1, len(imgs), 'Counting Images:', 'Done!', 2, 50)
    _id = id.split('_')[0]
    _info = result['image_info'][_id]

    _age = int(_info['age'])
    _gender = _info['gender']

    if _gender == "M":
        labels["M"].append(id)
    else:
        labels["F"].append(id)

    if _age >= 0 and _age <= 3:
        labels["0"].append(id)
    elif _age >= 4 and _age <= 7:
        labels["1"].append(id)
    elif _age >= 8 and _age <= 13:
        labels["2"].append(id)
    elif _age >= 14 and _age <= 20:
        labels["3"].append(id)
    elif _age >= 21 and _age <= 33:
        labels["4"].append(id)
    elif _age >= 34 and _age <= 45:
        labels["5"].append(id)
    elif _age >= 46 and _age <= 59:
        labels["6"].append(id)
    elif _age >= 60:
        labels["7"].append(id)


def divider(t):
    _test = int(t * _test_ratio)
    _valid = int(t * _valid_ratio)
    _train = t - _test - _valid

    return _train, _valid, _test

for label in labels:
    # printProgress(i + 1, len(imgs), 'Processing Images:', 'Done!', 2, 50)

    ids = list(set(labels[label]))
    shuffle(ids)
    _ids = [id.split('_', 1)[0] for id in ids]
    _total = len(ids)

    print(label, _total)

    _train, _valid, _test = divider(_total)
    print('Train:',_train,'Valid:', _valid,'Test:', _test)

    if label == "M" or label == "F":
        result['training_ids']['gender'] = result['training_ids']['gender'] + _ids[0:_train]
        result['validation_ids']['gender'] = result['validation_ids']['gender'] + _ids[_train:_valid]
        result['test_ids']['gender'] = result['test_ids']['gender'] + _ids[_valid:_test]

        for i, img in enumerate(ids):
            printProgress(i + 1, len(ids), 'Processing Images for Label '+label+':', 'Done!', 2, 50)
            if i >= 0 and i < _train:
                image_process(img,result['training_ids']['gender'])
            elif i >= _train and i < _valid:
                image_process(img,result['validation_ids']['gender'])
            elif i >= _valid and i < _test:
                image_process(img,result['test_ids']['gender'])
    else:
        result['training_ids']['age'] = result['training_ids']['age'] + _ids[0:_train]
        result['validation_ids']['age'] = result['validation_ids']['age'] + _ids[_train:_valid]
        result['test_ids']['age'] = result['test_ids']['age'] + _ids[_valid:_test]
        for i, img in enumerate(ids):
            printProgress(i + 1, len(ids), 'Processing Images for Label '+label+':', 'Done!', 2, 50)
            if i >= 0 and i < _train:
                image_process(img,result['training_ids']['age'])
            elif i >= _train and i < _valid:
                image_process(img,result['validation_ids']['age'])
            elif i >= _valid and i < _test:
                image_process(img,result['test_ids']['age'])



print('Saving dict to json file...')
with open('annotation.json', 'w') as outfile:
    json.dump(result, outfile)