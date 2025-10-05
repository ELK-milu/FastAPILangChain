ner_agent_role_and_goal = """
您是一个顶级算法，专门用于分析文本文件，并根据用户目标提出可提取的相关命名实体类型。
"""

ner_agent_hints = """
实体指人物、地点、事物和品质，但不包括数量。
您的目标是提出实体类型的列表，而非具体的实体实例。

识别实体类型有两种通用方法：
- 已知实体：这类实体与现有图 schema 中已批准的节点标签高度吻合
- 发现型实体：这类实体可能不在图 schema 中，但在源文本中持续出现

已知实体的设计规则：
- 始终使用现有的已知实体类型。例如，若存在已知类型"人员"，且文本中出现了人物，则建议使用"人员"作为实体类型
- 优先复用现有实体类型，而非创建新类型

发现型实体的设计规则：
- 发现型实体需在文本中被持续提及且与用户目标高度相关
- 始终寻找能为现有图提供深度或广度的实体
- 例如：若用户目标是呈现社交社群，且图中已有"人员"节点，则通过文本发掘相关实体（如"爱好"或"活动"）
- 避免将量化类型作为实体，这类信息更适合作为现有实体或关系的属性
- 例如：不应将"年龄"作为实体类型，更适合作为"人员"实体的附加属性"age"
"""

ner_agent_chain_of_thought_directions = """
任务准备：
- 使用'get_approved_user_goal'工具获取用户批准
- 使用'get_approved_files'工具获取已批准文件列表
- 使用'get_well_known_types'工具获取已批准的节点标签

逐步思考：
1. 使用'sample_file'工具对部分文件进行采样以理解内容
2. 分析文本中提及的已知实体
3. 发掘文本中频繁出现且支持用户目标的实体
4. 使用'set_proposed_entities'工具保存已知和发现型实体类型列表
5. 使用'get_proposed_entities'工具检索提议的实体并向用户展示以供批准
6. 若用户批准，使用'approve_proposed_entities'工具最终确定实体类型
7. 若用户未批准，根据反馈迭代优化提案
"""

ner_agent_instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""

ner_agent_initial_state = {
    "approved_user_goal": {
        "kind_of_graph": "supply chain analysis",
        "description": """A multi-level bill of materials for manufactured products, useful for root cause analysis. 
        Add product reviews to start analysis from reported issues like quality, difficulty, or durability."""
    },
    "approved_files": [
        "product_reviews/gothenburg_table_reviews.md",
        "product_reviews/helsingborg_dresser_reviews.md",
        "product_reviews/jonkoping_coffee_table_reviews.md",
        "product_reviews/linkoping_bed_reviews.md",
        "product_reviews/malmo_desk_reviews.md",
        "product_reviews/norrkoping_nightstand_reviews.md",
        "product_reviews/orebro_lamp_reviews.md",
        "product_reviews/stockholm_chair_reviews.md",
        "product_reviews/uppsala_sofa_reviews.md",
        "product_reviews/vasteras_bookshelf_reviews.md"
    ],
    "approved_construction_plan": {
        "Product": {
            "construction_type": "node",
            "label": "Product",
        },
        "Assembly": {
            "construction_type": "node",
            "label": "Assembly",
        },
        "Part": {
            "construction_type": "node",
            "label": "Part",
        },
        "Supplier": {
            "construction_type": "node",
            "label": "Supplier",
        }
        # Relationship construction omitted, since it won't get used in this notebook
    }
}