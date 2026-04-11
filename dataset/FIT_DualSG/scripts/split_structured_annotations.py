#!/usr/bin/env python3
"""
将结构化annotations拆分为4个不同视角的JSON文件
"""

import json
import copy

def split_annotations(input_file, output_dir):
    """
    拆分结构化annotations为4个不同视角的文件
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建4个视角的数据
    global_structural = []
    dynamic_behavior = []
    event_centric = []
    semantic_caption = []
    
    for item in data:
        # 解析annotations字段
        annotations_str = item.get('annotations', '{}')
        
        # 尝试解析JSON
        try:
            if isinstance(annotations_str, str):
                annotations = json.loads(annotations_str)
            else:
                annotations = annotations_str
        except:
            annotations = {}
        
        # 1. Global Structural View（全局结构）
        global_item = copy.deepcopy(item)
        global_annotations = {}
        if 'overall_trend' in annotations:
            global_annotations['overall_trend'] = annotations['overall_trend']
        if 'recent_regime' in annotations:
            global_annotations['recent_regime'] = annotations['recent_regime']
        global_item['annotations'] = json.dumps(global_annotations)
        global_structural.append(global_item)
        
        # 2. Dynamic Behavior View（动态行为）
        dynamic_item = copy.deepcopy(item)
        dynamic_annotations = {}
        if 'volatility' in annotations:
            dynamic_annotations['volatility'] = annotations['volatility']
        if 'volatile_segments' in annotations:
            dynamic_annotations['volatile_segments'] = annotations['volatile_segments']
        dynamic_item['annotations'] = json.dumps(dynamic_annotations)
        dynamic_behavior.append(dynamic_item)
        
        # 3. Event-centric View（事件视角，最重要）
        event_item = copy.deepcopy(item)
        event_annotations = {}
        if 'major_turning_points' in annotations:
            event_annotations['major_turning_points'] = annotations['major_turning_points']
        event_item['annotations'] = json.dumps(event_annotations)
        event_centric.append(event_item)
        
        # 4. Semantic Caption View（弱语义）
        semantic_item = copy.deepcopy(item)
        semantic_annotations = {}
        if 'caption' in annotations:
            semantic_annotations['caption'] = annotations['caption']
        semantic_item['annotations'] = json.dumps(semantic_annotations)
        semantic_caption.append(semantic_item)
    
    # 保存4个文件
    files = {
        'fit_dualsg_global_structural.json': global_structural,
        'fit_dualsg_dynamic_behavior.json': dynamic_behavior,
        'fit_dualsg_event_centric.json': event_centric,
        'fit_dualsg_semantic_caption.json': semantic_caption
    }
    
    for filename, content in files.items():
        output_path = f'{output_dir}/{filename}'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f'✓ 保存文件: {output_path} (样本数: {len(content)})')
    
    print(f'\n✓ 完成！共生成4个文件')

if __name__ == '__main__':
    input_file = './dataset/FIT_DualSG/fit_dualsg_structured.json'
    output_dir = './dataset/FIT_DualSG'
    
    split_annotations(input_file, output_dir)
