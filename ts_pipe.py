#!/usr/bin/env python3
import fire

if __name__ == '__main__':
    from ts.datasets.pipe import DiabetesPipeline
    fire.Fire(DiabetesPipeline)