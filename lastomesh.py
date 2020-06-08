
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


def get_metrics(nparray):
    return {
        'mean': np.mean(nparray),
        'median': np.median(nparray),
        'max': np.max(nparray),
        'min': np.min(nparray),
        'var': np.var(nparray),
    }


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
        r = requests.get(self.url, stream=True)
        if r.status_code == 200:
            with self.output().open('w') as f:
                # f.write(r.content)
                for chunk in r.iter_content(chunk_size=1024*10):
                    f.write(chunk)
        else:
            raise ConnectionError(r.status_code)


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
        subprocess.check_output(cmd)


class DownloadShizuokaPCD(luigi.Task):
    product_id = luigi.Parameter()
    base_url = 'https://raw.githubusercontent.com/colspan/pcd-open-datasets/master/shizuokapcd/product/{}.json'
    output_dir = luigi.Parameter(default='tmp/mesh')
    work_dir = luigi.Parameter(default='tmp/work')

    def requires(self):
        return TextDownloader(
            url=self.base_url.format(self.product_id),
            filepath=os.path.join(self.work_dir, '{}.json'.format(self.product_id)))

    def output(self):
        return {
            'stat_info': luigi.LocalTarget(os.path.join(
                self.output_dir,
                'stat-{}.json'.format(self.product_id)),
                format=luigi.format.UTF8),
        }

    def load_product_info(self):
        with self.input().open('r') as f:
            product_info = json.load(f)
        return product_info

    def load_stat_info(self):
        with self.output()['stat_info'].open('r') as f:
            stat_info = json.load(f)
        return stat_info

    def load_dataset(self, skip_rate=0):
        download_tasks = self.download_tasks()
        lasdataset = []
        for download_task in download_tasks:
            lasdataset.append(
                LasFile(download_task.output().path).toarray(skip_rate=skip_rate))

        lasdata = np.concatenate(lasdataset)
        return lasdata

    def download_tasks(self):
        # load metadata
        product_info = self.load_product_info()
        las_urls = product_info['lasUrls']['value']
        download_tasks = [
            BinaryDownloader(
                url=x,
                filepath=os.path.join(self.work_dir, os.path.basename(x)))
            for x in las_urls
        ]
        return download_tasks

    def run(self):
        # load metadata
        product_info = self.load_product_info()

        # get las dataset
        download_tasks = self.download_tasks()
        yield download_tasks

        # load dataset
        skip_rate = 0.8
        lasdataset = []
        for download_task in download_tasks:
            lasdataset.append(
                LasFile(download_task.output().path).toarray(skip_rate=skip_rate))

        lasdata = np.concatenate(lasdataset)
        lasdata_shape = lasdata.shape
        if skip_rate > 0:
            lasdata_shape = (
                int(float(lasdata_shape[0]) / (1.0 - skip_rate)), lasdata_shape[1])

        plydata = PlyFile(data=lasdata)
        pcd = plydata.obj

        distances = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(
            pcd)
        stat_info = {
            'productIdFlat': {
                'value': self.product_id,
            },
            'shape': {
                'value': lasdata_shape,
            },
            'metrics': {
                'value': {
                    'distance': get_metrics(distances),
                    'x': get_metrics(lasdata[:, 0]),
                    'y': get_metrics(lasdata[:, 1]),
                    'z': get_metrics(lasdata[:, 2]),
                    'i': get_metrics(lasdata[:, 3]),
                    'r': get_metrics(lasdata[:, 4]),
                    'g': get_metrics(lasdata[:, 5]),
                    'b': get_metrics(lasdata[:, 6]),
                },
            },
            'downloadedFiles': {
                'value': [x.output().path for x in download_tasks],
            },
        }
        product_info.update(stat_info)

        with self.output()['stat_info'].open('w') as f:
            json.dump(product_info, f, indent=2, ensure_ascii=False)


class CreateMeshFromLasData(luigi.Task):
    product_id = luigi.Parameter()
    base_url = 'https://raw.githubusercontent.com/colspan/pcd-open-datasets/master/shizuokapcd/product/{}.json'
    output_dir = luigi.Parameter(default='tmp/mesh')
    output_filename = luigi.Parameter(default=None)
    work_dir = luigi.Parameter(default='tmp/work')
    file_format = luigi.Parameter(default='ply')
    mesh_type = luigi.Parameter(default='poisson')
    simplify_type = luigi.Parameter(default=None)
    skip_meshing = luigi.Parameter(default=False)

    def requires(self):
        return DownloadShizuokaPCD(product_id=self.product_id,
                                   work_dir=self.work_dir,
                                   output_dir=self.output_dir)

    def output(self):
        output_filename = 'mesh-{}.{}'.format(self.product_id, self.file_format) \
            if self.output_filename is None \
            else self.output_filename
        return {
            'mesh_file': luigi.LocalTarget(os.path.join(
                self.output_dir, output_filename)),
        }

    def run(self):
        stat_info = self.requires().load_stat_info()
        print(stat_info)

        if stat_info['shape']['value'][0] > 100000000:
            skip_rate = 0.5
        else:
            skip_rate = 0

        lasdata = self.requires().load_dataset(skip_rate=skip_rate)
        plydata = PlyFile(data=lasdata)
        pcd = plydata.obj

        # 指定したvoxelサイズでダウンサンプリング
        print('downsizing')
        avg_dist = stat_info['metrics']['value']['distance']['mean']
        voxel_size = avg_dist * 3
        voxel_down_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            pcd, voxel_size=voxel_size)
        target_pcd = voxel_down_pcd
        pcd_center = target_pcd.get_center().tolist()

        if self.skip_meshing:
            # データ保存
            o3d.io.write_point_cloud(self.output()['mesh_file'].path, target_pcd)
            return

        # 法線計算
        print('estimate normal vectors')
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size,
                max_nn=30))
        # target_pcd, _ = target_pcd.remove_statistical_outlier(5, 1.5)
        # target_pcd.orient_normals_to_align_with_direction()
        target_pcd.orient_normals_towards_camera_location(
            pcd_center[:-1]+[pcd_center[-1]*100])
        target_pcd = target_pcd.normalize_normals()

        # メッシュ化
        print('meshing')
        if self.mesh_type == 'ball-pivoting':
            radius = voxel_size
            radii = [
                radius*0.5,
                radius,
                radius*2,
                radius*4,
                radius*8,
                radius*16,
            ]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                target_pcd, o3d.utility.DoubleVector(radii))
        else:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                target_pcd, depth=11, linear_fit=True)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        print('simplifying meshes')
        # TODO reduce memory usage
        if self.simplify_type == 'quadric-decimation':
            mesh = mesh.simplify_quadric_decimation(
                int(len(mesh.triangles)*0.05))
        elif self.simplify_type == 'vertex-clustering':
            mesh = mesh.simplify_vertex_clustering(voxel_size*5)
            mesh = mesh.simplify_quadric_decimation(
                int(len(mesh.triangles)*0.5))
        else:
            pass

        # データ保存
        o3d.io.write_triangle_mesh(self.output()['mesh_file'].path, mesh)


# class RenderPointCloud(luigi.Task):
#     product_id = luigi.Parameter()
#     output_dir = luigi.Parameter(default='tmp/mesh')
#     output_filename = luigi.Parameter(default=None)
#     work_dir = luigi.Parameter(default='tmp/work')
#     file_format = luigi.Parameter(default='ply')
#     mesh_type = luigi.Parameter(default='poisson')
#     simplify_type = luigi.Parameter(default=None)

#     def requires(self):
#         return CreateMeshFromLasData(
#             product_id=self.product_id,
#             output_dir=self.output_dir,
#             output_filename=self.output_filename,
#             work_dir=self.work_dir,
#             file_format=self.file_format,
#             mesh_type=self.mesh_type,
#             simplify_type=self.simplify_type)

#     def output(self):
#         return luigi.LocalTarget(
#             os.path.join(self.output_dir,
#                          '{}.png'.format(self.output_filename)))

#     def run(self):
#         mesh = o3d.io.read_triangle_mesh(self.input().path)
#         # メッシュデータの表示
#         o3d.visualization.draw_geometries([mesh])


class ShowPointCloud(luigi.Task):
    product_id = luigi.Parameter()

    def requires(self):
        return CreateMeshFromLasData(self.product_id)

    def run(self):
        mesh = o3d.io.read_triangle_mesh(self.input()['mesh_file'].path)
        # メッシュデータの表示
        o3d.visualization.draw_geometries([mesh])


class DownloadShizuokaPCDs(luigi.WrapperTask):
    product_list = luigi.Parameter()
    output_dir = luigi.Parameter(default='tmp/mesh')
    work_dir = luigi.Parameter(default='tmp/work')

    def requires(self):
        with open(self.product_list, 'r') as f:
            product_list = json.load(f)
        return [
            DownloadShizuokaPCD(
                product_id=product['id'],
                work_dir=os.path.join(self.work_dir, product['id']),
                output_dir=self.output_dir)
            for product in product_list
        ]


if __name__ == "__main__":
    luigi.run()
