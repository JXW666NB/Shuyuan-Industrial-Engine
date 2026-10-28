# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/home/featurize/tgai/training/tests")
import test_train_api as t
print("A", t.test_autograd())
print("C", t.test_muon())
