
#!/bin/env python
# -*- coding: utf-8 -*-
import json
import luigi
import os
import requests
import subprocess

import numpy as np
import open3d as o3d
from lasto3dtiles.format.las import LasFile
from lasto3dtiles.format.ply import PlyFile


class TextDownloader(luigi.Task):
    filepath = luigi.Parameter()
    url = luigi.Parameter()
    decode = luigi.Parameter(default='utf-8')

    def output(self):
        return luigi.LocalTarget(
            format=luigi.format.UTF8, path=self.filepath)

    def run(self):
        r = requests.get(self.url)
        with self.output().open('w') as f:
            f.write(r.content.decode(self.decode))


class BinaryDownloader(luigi.Task):
    filepath = luigi.Parameter()
    url = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            format=luigi.format.Nop, path=self.filepath)

    def run(self):
        r = requests.get(self.url)
        with self.output().open('w') as f:
            f.write(r.content)


class ConvertLasFile(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(
            format=luigi.format.Nop, path=self.output_path)

    def run(self):
        cmd = [
            'las2las',
            '-f',
            '1.0',
            self.input_path,
            self.output_path,
        ]
        subprocess.call(cmd)


class CreateMeshFromLasData(luigi.Task):
    product_id = luigi.Parameter()
    base_url = 'https://raw.githubusercontent.com/colspan/pcd-open-datasets/master/shizuokapcd/product/{}.json'

    def requires(self):
        return TextDownloader(
            url=self.base_url.format(self.product_id),
            filepath=os.path.join('tmp', '{}.json'.format(self.product_id)))

    def output(self):
        return luigi.LocalTarget(os.path.join(
            'tmp', '{}.ply'.format(self.product_id)))

    def run(self):
        # load metadata
        with self.input().open('r') as f:
            product_info = json.load(f)
        las_urls = product_info['lasUrls']['value']

        # get las dataset
        download_tasks = [
            BinaryDownloader(
                url=x,
                filepath=os.path.join('tmp', os.path.basename(x)))
            for x in las_urls
        ]
        yield download_tasks

        # convert las files
        convert_tasks = []
        for download_task in download_tasks:
            original_path = os.path.basename(download_task.output().path)
            converted_path = os.path.join(
                'tmp', 'converted-{}'.format(os.path.basename(original_path)))
            convert_tasks.append(
                ConvertLasFile(
                    input_path=original_path,
                    output_path=converted_path
                ))
        yield convert_tasks

        #
        lasdataset = []
        for convert_task in convert_tasks:
            lasdataset.append(
                LasFile(convert_task.output().path).toarray(skip_rate=0.5))

        lasdata = np.concatenate(lasdataset)
        # lasdata = lasdataset[1]

        print(lasdata.shape)
        plydata = PlyFile(data=lasdata)
        pcd = plydata.obj

        # 指定したvoxelサイズでダウンサンプリング
        voxel_down_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            pcd, voxel_size=0.1)

        # 法線計算
        o3d.geometry.PointCloud.estimate_normals(voxel_down_pcd)

        distances = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(
            voxel_down_pcd)
        avg_dist = np.mean(distances)
        print(avg_dist)

        # メッシュ化
        radius = 3 * avg_dist
        radii = [radius, radius * 2]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            voxel_down_pcd, o3d.utility.DoubleVector(radii))

        # データ保存
        o3d.io.write_triangle_mesh(self.output().path, mesh)


class ShowPointCloud(luigi.Task):
    product_id = luigi.Parameter()

    def requires(self):
        return CreateMeshFromLasData(self.product_id)

    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.input().path)
        # メッシュデータの表示
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    luigi.run()
