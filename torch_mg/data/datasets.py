from torch.utils.data import Dataset
import itertools


class TXT_Dataset(Dataset):
    """Dataset representing text files"""

    def __init__(self, file_name, filter_fn=lambda _x:False,
                 preprocess_fn=lambda _x:_x.strip('\n').strip('\r'),
                 num_folds=10, fold_id=0, is_train=False):
        self.file_name = file_name
        self.filter_fn = filter_fn
        self.preprocess_fn = preprocess_fn
        self.num_folds = num_folds
        self.fold_id = fold_id
        self.is_train = is_train

        # read dataset
        with open(file_name) as f:
            lines = [preprocess_fn(line) if preprocess_fn is not None else line
                     for line in f if line != '' and not filter_fn(line)]

        if num_folds <= 1:
            self.lines = lines
        else:
            chunk_size = int(float(len(lines))/num_folds)
            chunks = []
            for i in range(num_folds):
                if i < num_folds - 1:
                    chunks.append(lines[i * chunk_size:(i + 1)*chunk_size])
                else:
                    chunks.append(lines[i * chunk_size:])

            if is_train:
                del chunks[fold_id]
                self.lines = list(itertools.chain(*chunks))
            else:
                self.lines = chunks[fold_id]

    def __getitem__(self, index):
        return self.lines[index]

    def __len__(self):
        return len(self.lines)

