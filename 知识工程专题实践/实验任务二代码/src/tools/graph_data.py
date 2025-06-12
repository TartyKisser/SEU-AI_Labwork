from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd


def homo_data(
        root,
        fill_mode="default",
        transform=None,
        return_id_mapping=False,
        **kwargs,
):
    # Load parcel features
    parcel_df = pd.read_csv(root + "spatial_join_result.csv", sep=",", low_memory=False)

    # Rename columns for clarity
    parcel_df = parcel_df.rename(columns={'OBJECTID_1': 'parcel_id'})

    # Extract the target variable (land use description)
    land_use_labels = parcel_df['LU_DESC'].unique()
    land_use_mapping = {lu: i for i, lu in enumerate(land_use_labels)}

    # Fill missing data if needed
    _fill_data(parcel_df, fill_mode, **kwargs)

    # Load connections data - both road and railway networks
    road_connections_df = pd.read_csv(root + "BUSWAY.csv", sep=",", low_memory=False)
    railway_connections_df = pd.read_csv(root + "SUBWAY.csv", sep=",", low_memory=False)

    # Add a network type indicator to distinguish between road and railway connections
    road_connections_df['network_type'] = 'road'
    railway_connections_df['network_type'] = 'railway'

    # Combine both networks
    connections_df = pd.concat([road_connections_df, railway_connections_df], ignore_index=True)

    # Select feature columns, excluding non-numeric and identifier columns
    feature_cols = ['Shape_Area', 'YDR', 'bus_station_counts', 'subway_station_counts',
                    'crossing_counts', 'catering', 'shopping', 'public_service',
                    'transportation', 'leisure']
    num_features = len(feature_cols)

    # Map parcel_id to node_id for the graph
    parcel_ids = parcel_df['parcel_id'].unique()
    num_parcels = len(parcel_ids)
    parcel_id_to_node = {pid: i for i, pid in enumerate(parcel_ids)}

    # 创建与原始函数相似的id_mapping结构
    id_mapping = {
        'parcel': parcel_id_to_node,
        'landuse': {lu: num_parcels + i for i, lu in enumerate(land_use_labels)}
    }

    # 创建node_labels，类似于原始函数
    node_labels = {'parcel': 0, 'landuse': 1}

    # 创建节点特征
    # 地块节点使用实际特征，地块用途节点使用零向量
    x_tensor = torch.tensor(
        np.vstack([
            parcel_df[feature_cols].fillna(0).to_numpy(),  # 地块特征
            np.zeros((len(land_use_labels), num_features))  # 地块用途类型特征（零向量）
        ]),
        dtype=torch.float,
    )

    # 创建节点类型标签
    y_tensor = torch.cat((
        torch.full(size=(num_parcels,), fill_value=node_labels['parcel']),  # 所有地块节点的标签为0
        torch.full(size=(len(land_use_labels),), fill_value=node_labels['landuse'])  # 所有地块用途节点的标签为1
    ))

    # 为每个节点分配其地块用途类型
    # 这个不会存储在y_tensor中，但会添加到Data对象中作为属性
    parcel_types = torch.tensor([land_use_mapping[lu] for lu in parcel_df['LU_DESC'].values], dtype=torch.long)

    # 创建边索引从连接数据
    edge_index = []
    edge_type = []  # 新增边类型标识，0表示公路，1表示铁路

    for _, row in connections_df.iterrows():
        if row['parcel_a'] in parcel_id_to_node and row['parcel_b'] in parcel_id_to_node:
            # 添加双向边（无向图）
            edge_index.append([parcel_id_to_node[row['parcel_a']], parcel_id_to_node[row['parcel_b']]])
            edge_index.append([parcel_id_to_node[row['parcel_b']], parcel_id_to_node[row['parcel_a']]])

            # 添加边类型
            edge_type_value = 0 if row['network_type'] == 'road' else 1
            edge_type.append(edge_type_value)
            edge_type.append(edge_type_value)  # 为反向边添加相同的类型

    # 还要添加地块到其用途类型的边
    parcel_to_landuse_start_idx = len(edge_index)  # 记录地块到用途边的起始索引
    for i, row in parcel_df.iterrows():
        parcel_idx = parcel_id_to_node[row['parcel_id']]
        landuse_idx = id_mapping['landuse'][row['LU_DESC']]
        edge_index.append([parcel_idx, landuse_idx])
        edge_index.append([landuse_idx, parcel_idx])  # 双向连接

        # 添加边类型，2表示地块到用途类型的边
        edge_type.append(2)
        edge_type.append(2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # 创建边属性（距离）
    edge_attr = []
    # 地块间连接的边属性（公路和铁路）
    for _, row in connections_df.iterrows():
        if row['parcel_a'] in parcel_id_to_node and row['parcel_b'] in parcel_id_to_node:
            # 添加距离作为边属性（双向）
            edge_attr.append([row['distance']])
            edge_attr.append([row['distance']])

    # 为地块到用途类型的边添加属性（使用默认值1.0表示关联强度）
    for _ in range(2 * parcel_df.shape[0]):  # 双向边的数量
        edge_attr.append([1.0])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 创建数据对象
    data = Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,  # 添加边类型信息
        node_labels=node_labels,  # 添加node_labels
        land_use_mapping=land_use_mapping,  # 保留land_use_mapping
        parcel_types=parcel_types,  # 添加地块类型信息
        parcel_to_landuse_start_idx=parcel_to_landuse_start_idx  # 添加地块到用途边的起始索引
    )

    if transform is not None:
        data = transform(data)

    if return_id_mapping:
        return data, id_mapping

    return data


# Helper function to fill missing data (simplified from original)
def _fill_data(df, fill_mode, **kwargs):
    if fill_mode == "default":
        # Fill numeric columns with 0 and categorical with the most frequent value
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    elif fill_mode == "mean":
        # Fill numeric columns with mean and categorical with most frequent
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    # Add more fill modes as needed