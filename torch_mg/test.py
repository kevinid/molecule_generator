import data
import models
import meta
from torch.optim import Adam
import gc

def test():
    # construct dataset and loaders

    db_train = data.TXT_Dataset('datasets/ChEMBL.txt', num_folds=5, fold_id=0, is_train=True)
    db_test = data.TXT_Dataset('datasets/ChEMBL.txt', num_folds=5, fold_id=0, is_train=False)

    loader_train = data.GraphLoader(db_train, batch_size=70, num_workers=4, k=5, p=0.8)
    loader_test = data.GraphLoader(db_test, batch_size=70, num_workers=0, k=5, p=0.8)

    it_train, it_test = iter(loader_train), iter(loader_test)

    # build model
    model = models.VanillaMolGen(len(meta.ATOM_TYPES), len(meta.BOND_TYPES), D=2,
                                 F_e=16, F_h=[32, 64, 128, 128, 256, 256], F_skip=512,
                                 F_c=[512, ], F_h_policy=256, k_softmax=1, memory_efficient=True)
    model_wrapper = models.DataParallel_Explicit(model, gather_dim=0).cuda()

    # construct optimizer
    optimizer = Adam(model.parameters(), lr=0.0005)

    for i in range(10):
        model_wrapper.zero_grad()
        optimizer.zero_grad()

        try:
            inputs = [next(it_train) for _ in range(4)]
        except StopIteration:
            it_train = iter(loader_train)
            inputs = [next(it_train) for _ in range(4)]

        # move to gpu
        inputs = [data.GraphLoader.from_numpy_to_tensor(input_i, i)[:-2]
                  for i, input_i in enumerate(inputs)]

        loss = model_wrapper(*inputs, mode='loss').mean()

        loss.backward()
        optimizer.step()

        print(loss)

        del inputs, loss
        gc.collect()

if __name__ == '__main__':
    test()
