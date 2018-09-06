
# word2vec常用python操作
## 导入word2vec moxing 
self.w2vcModel = KeyedVectors.load_word2vec_format(os.path.join(config.BASH_PATH, 'lstm_data', 'news12g_bdbk20g_nov90g_dim128.bin'), binary=True)