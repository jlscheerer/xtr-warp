from colbert.utils.tracker import NOPTracker


class IndexLoaderWARP:
    def __init__(self, index_path, use_gpu=True, load_index_with_mmap=False):
        assert not use_gpu
        assert not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap

        raise NotImplementedError()


class IndexScorerWARP(IndexLoaderWARP):
    def __init__(self, index_path, use_gpu=True, load_index_with_mmap=False):
        assert not use_gpu
        assert not load_index_with_mmap

        super().__init__(
            index_path=index_path,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
        )

        IndexScorerWARP.try_load_torch_extensions(use_gpu)

        # self.ivf_strided = StridedTensor(
        #     self.codes_compacted, self.sizes_compacted, use_gpu=self.use_gpu
        # )

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        raise NotImplementedError()

        cls.loaded_extensions = True

    def rank(
        self,
        config,
        Q,
        filter_fn=None,
        pids=None,
        tracker=NOPTracker(),
    ):
        assert filter_fn is None
        assert pids is None

        raise NotImplementedError()
