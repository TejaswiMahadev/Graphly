import gradio as gr
import pandas as pd
import networkx as nx
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from neo4j import GraphDatabase
import plotly.graph_objects as go
import numpy as np

class GlobalState:
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.neo4j_driver = None

state = GlobalState()

def extract_knowledge_graph(text, api_key):
    if not api_key:
        return "Please enter your Google API key to use the AI extraction feature.", None, None, None
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    template = """
    Your task is to extract entities and relationships from the text below to create a knowledge graph.
    
    Text: {text}
    
    STRICT REQUIREMENTS - FOLLOW EXACTLY:
    1. Output a valid JSON object with exactly two keys: "nodes" and "relationships"
    2. EVERY node MUST have BOTH "id" and "label" fields:
       - "id": A unique identifier string for the entity (e.g., "Zeus", "Apple", "Steve_Jobs")
       - "label": The entity type or category (e.g., "Person", "Company", "Location", "Concept")
    3. EVERY relationship MUST have "source", "target", and "type" fields:
       - "source": Must exactly match the "id" of an existing node
       - "target": Must exactly match the "id" of an existing node
       - "type": Should be in ALL_CAPS with underscores (e.g., "FOUNDED", "MARRIED_TO", "WORKS_FOR")
    
    EXAMPLE FORMAT:
    {{
        "nodes": [
            {{"id": "Zeus", "label": "God"}},
            {{"id": "Mount_Olympus", "label": "Location"}},
            {{"id": "Hera", "label": "Goddess"}}
        ],
        "relationships": [
            {{"source": "Zeus", "target": "Mount_Olympus", "type": "RULES_FROM"}},
            {{"source": "Zeus", "target": "Hera", "type": "MARRIED_TO"}}
        ]
    }}
    
    IMPORTANT RULES:
    - Every entity mentioned in relationships MUST have a corresponding node definition
    - Ensure consistent capitalization in IDs across nodes and relationships
    - No duplicate node IDs allowed
    - Use clear, descriptive category labels (e.g., "Person", "Organization", "Location")
    - Relationship types should be meaningful verbs or connections
    
    Output ONLY valid JSON with no explanations, comments, or markdown formatting.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.invoke({"text": text})
        result_text = result['text']
        import re
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", result_text)
        if json_match:
            result_text = json_match.group(1)
        else:
            json_match = re.search(r"(\{[\s\S]*\})", result_text)
            if json_match:
                result_text = json_match.group(1)
        result_text = result_text.strip()
        try:
            kg_data = json.loads(result_text)
            if 'nodes' not in kg_data:
                kg_data['nodes'] = []
            if 'relationships' not in kg_data:
                kg_data['relationships'] = []
                
            extracted_nodes = kg_data['nodes']
            extracted_relationships = kg_data['relationships']
            node_ids = set()
            fixed_nodes = []
            for i, node in enumerate(extracted_nodes):
                fixed_node = {}
                if 'id' not in node or not node['id']:
                    fixed_node['id'] = f"Entity_{i+1}"
                else:
                    fixed_node['id'] = str(node['id']).replace(' ', '_')
                if 'label' not in node or not node['label']:
                    fixed_node['label'] = "Entity"
                else:
                    fixed_node['label'] = str(node['label'])
                node_ids.add(fixed_node['id'])
                fixed_nodes.append(fixed_node)
            fixed_relationships = []
            for i, rel in enumerate(extracted_relationships):
                fixed_rel = {}
                if 'source' not in rel or not rel['source']:
                    continue  
                if 'target' not in rel or not rel['target']:
                    continue  
                if 'type' not in rel or not rel['type']:
                    rel['type'] = "RELATED_TO"
                fixed_rel['source'] = str(rel['source']).replace(' ', '_')
                fixed_rel['target'] = str(rel['target']).replace(' ', '_')
                fixed_rel['type'] = str(rel['type']).upper().replace(' ', '_')
                if fixed_rel['source'] not in node_ids:
                    new_node = {"id": fixed_rel['source'], "label": "Entity"}
                    fixed_nodes.append(new_node)
                    node_ids.add(fixed_rel['source'])
                
                if fixed_rel['target'] not in node_ids:
                    new_node = {"id": fixed_rel['target'], "label": "Entity"}
                    fixed_nodes.append(new_node)
                    node_ids.add(fixed_rel['target'])
                
                fixed_relationships.append(fixed_rel)
            existing_node_ids = {node['id'] for node in state.nodes}
            for node in fixed_nodes:
                if node['id'] not in existing_node_ids:
                    state.nodes.append(node)
                    existing_node_ids.add(node['id'])
            existing_rels = {(rel['source'], rel['target'], rel['type']) for rel in state.relationships}
            for rel in fixed_relationships:
                rel_key = (rel['source'], rel['target'], rel['type'])
                if rel_key not in existing_rels:
                    state.relationships.append(rel)
                    existing_rels.add(rel_key)
            nodes_df = pd.DataFrame(fixed_nodes)
            relationships_df = pd.DataFrame(fixed_relationships)
            fig = visualize_graph_3d()
            
            success_message = f"Successfully extracted {len(fixed_nodes)} nodes and {len(fixed_relationships)} relationships!"
            return success_message, nodes_df, relationships_df, fig
            
        except json.JSONDecodeError as e:
            error_message = f"Failed to parse AI response into JSON. Attempting direct extraction from text."
            import re
            potential_entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
            unique_entities = list(set(potential_entities))
            fallback_nodes = []
            for i, entity in enumerate(unique_entities):
                entity_id = entity.replace(' ', '_')
                label = "Person" if any(word in text.lower() for word in [" he ", " she ", " his ", " her "]) else "Entity"
                fallback_nodes.append({"id": entity_id, "label": label})
                state.nodes.append({"id": entity_id, "label": label})
            fallback_rels = []
            
            nodes_df = pd.DataFrame(fallback_nodes)
            relationships_df = pd.DataFrame(fallback_rels)
            fig = visualize_graph_3d()
            
            fallback_message = f"Used fallback extraction: found {len(fallback_nodes)} potential entities. Please review and edit them manually."
            return fallback_message, nodes_df, relationships_df, fig
            
    except Exception as e:
        error_message = f"Error during extraction: {str(e)}\n\nPlease try again or enter nodes manually."
        return error_message, None, None, None
    
def get_nodes_df():
    if not state.nodes:
        return None
    return pd.DataFrame(state.nodes)

def get_relationships_df():
    if not state.relationships:
        return None
    return pd.DataFrame(state.relationships)

def connect_to_neo4j(uri, username, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            session.run("MATCH (n) RETURN count(n) LIMIT 1")
        state.neo4j_driver = driver
        return "Connected to Neo4j successfully!"
    except Exception as e:
        return f"Failed to connect to Neo4j: {str(e)}"
    

def create_node(tx, node_id, node_label):
    tx.run(
        f"MERGE (n:{node_label} {{id: $id}}) RETURN n",
        id=node_id
    )

def create_relationship(tx, source_id, target_id, rel_type):
    tx.run(
        f"MATCH (a), (b) WHERE a.id = $source_id AND b.id = $target_id "
        f"MERGE (a)-[r:{rel_type}]->(b) RETURN r",
        source_id=source_id, target_id=target_id
    )

def save_to_neo4j():
    if not state.neo4j_driver:
        return "Please connect to Neo4j first."
    
    try:
        with state.neo4j_driver.session() as session:
            
            for node in state.nodes:
                session.execute_write(create_node, node['id'], node['label'])
            
            
            for rel in state.relationships:
                session.execute_write(create_relationship, rel['source'], rel['target'], rel['type'])
        return "Successfully saved to Neo4j!"
    except Exception as e:
        return f"Error saving to Neo4j: {str(e)}"

def add_node(node_id, node_label):
    if not node_id or not node_label:
        return "Please provide both ID and Label for the node.", None, None
    existing_node_ids = {node['id'] for node in state.nodes}
    if node_id in existing_node_ids:
        return f"Node with ID '{node_id}' already exists.", None, None
    
    state.nodes.append({"id": node_id, "label": node_label})
    nodes_df = pd.DataFrame(state.nodes)

    fig = visualize_graph_3d()
    
    return f"Added node: {node_id} ({node_label})", nodes_df, fig

def add_relationship(source_node, target_node, rel_type):
    if not source_node or not target_node or not rel_type:
        return "Please select source, target, and relationship type.", None, None
    rel_type = rel_type.upper()
    
    existing_rels = {(rel['source'], rel['target'], rel['type']) for rel in state.relationships}
    rel_key = (source_node, target_node, rel_type)
    
    if rel_key in existing_rels:
        return "This relationship already exists.", None, None
    
    state.relationships.append({
        "source": source_node,
        "target": target_node,
        "type": rel_type
    })
    
    relationships_df = pd.DataFrame(state.relationships)
    fig = visualize_graph_3d()
    
    return f"Added relationship: {source_node} --[{rel_type}]--> {target_node}", relationships_df, fig

def clear_data():
    state.nodes = []
    state.relationships = []
    fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode='markers')])
    fig.update_layout(title="Knowledge Graph (3D)")
    
    return "All data cleared!", None, None, fig

def export_csv():
    if not state.nodes or not state.relationships:
        return "No data to export.", None, None
    
    nodes_df = pd.DataFrame(state.nodes)
    relationships_df = pd.DataFrame(state.relationships)
    
    nodes_csv = nodes_df.to_csv(index=False)
    relationships_csv = relationships_df.to_csv(index=False)
    
    return "Data exported as CSV!", nodes_csv, relationships_csv

def visualize_graph_3d():
    if not state.nodes:
        fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode='markers')])
        fig.update_layout(
            title="The Knowledge Graph in 3D",
            font=dict(family="Times New Roman", size=14),
            scene=dict(
                bgcolor='rgba(240, 253, 255, 1)' 
            )
        )
        return fig
    G = nx.DiGraph()
    for node in state.nodes:
        G.add_node(node['id'], label=node['label'])
    for rel in state.relationships:
        G.add_edge(rel['source'], rel['target'], label=rel['type'])
    pos = nx.spring_layout(G, dim=3, seed=42)
    node_count = len(G.nodes())
    scale_factor = 1.5 
    np.random.seed(42)  
    for node in pos:
        pos[node] = np.array(pos[node]) * scale_factor
        pos[node] += np.random.uniform(-0.1, 0.1, 3)
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_sizes = []
    node_colors = []
    unique_labels = list(set(nx.get_node_attributes(G, 'label').values()))
    
    playful_colors = [
        '#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7', 
        '#C7CEEA', '#F8C8DC', '#BCE784', '#5DD39E', '#348AA7'
    ]
    
    color_map = {label: playful_colors[i % len(playful_colors)] for i, label in enumerate(unique_labels)}
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        label = G.nodes[node]['label']
        node_text.append(f"🔍 {node} ({label})")
        size = 10 + (G.degree(node) * 2)
        node_sizes.append(size)
        node_colors.append(color_map[label])
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.85,
            line=dict(color='white', width=1),
            symbol='circle'
        ),
        text=node_text,
        hoverinfo='text',
        name='Nodes'
    )
    edge_x = []
    edge_y = []
    edge_z = []
    edge_text = []
    edge_colors = []
    
    for edge in G.edges():
        source, target = edge
        x0, y0, z0 = pos[source]
        x1, y1, z1 = pos[target]

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        mid_z = (z0 + z1) / 2
        
        offset = np.random.uniform(0.05, 0.15)
        random_dir = np.random.uniform(-1, 1, 3)
        random_dir = random_dir / np.linalg.norm(random_dir) * offset
        
        mid_x += random_dir[0]
        mid_y += random_dir[1]
        mid_z += random_dir[2]
        
        edge_x.extend([x0, mid_x, x1, None])
        edge_y.extend([y0, mid_y, y1, None])
        edge_z.extend([z0, mid_z, z1, None])
        
        rel_type = G.edges[edge]['label']
        edge_text.append(f"✨ {source} --[{rel_type}]--> {target}")
        
        edge_colors.extend(['rgba(255, 140, 0, 0.7)'] * 4) 
    
   
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            color=edge_colors,
            width=3
        ),
        hoverinfo='text',
        text=edge_text,
        name='Relationships'
    )
    num_stars = 200
    star_range = 2.5  
    star_x = np.random.uniform(-star_range, star_range, num_stars)
    star_y = np.random.uniform(-star_range, star_range, num_stars)
    star_z = np.random.uniform(-star_range, star_range, num_stars)
    
    star_trace = go.Scatter3d(
        x=star_x, y=star_y, z=star_z,
        mode='markers',
        marker=dict(
            size=1.5,
            color='rgba(255, 255, 255, 0.8)',
            opacity=0.6
        ),
        hoverinfo='none',
        name='Stars'
    )
    
    fig = go.Figure(data=[star_trace, edge_trace, node_trace])
    
    fig.update_layout(
        title={
            'text': "✨ Knowledge Graph Explorer ✨",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'family': 'Arial', 'color': '#4B0082'}
        },
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
            bgcolor='rgba(5, 10, 30, 1)'  # Dark blue/black background like space
        )
    )
    
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig

def get_node_ids():
    return [node['id'] for node in state.nodes]

def update_dropdowns():
    choices = get_node_ids()
    return gr.Dropdown.update(choices=choices), gr.Dropdown.update(choices=choices)

def refresh_displays():
    nodes_df = get_nodes_df()
    rels_df = get_relationships_df()
    graph = visualize_graph_3d()
    dropdown_choices = get_node_ids()
    
    return (
        nodes_df, 
        rels_df, 
        graph,
        gr.Dropdown.update(choices=dropdown_choices),
        gr.Dropdown.update(choices=dropdown_choices)
    )

with gr.Blocks(title="3D Knowledge Graph Builder") as app:
    gr.Markdown("# 3D Knowledge Graph Builder")
    nodes_df = gr.DataFrame(label="Nodes", visible=False)
    rels_df = gr.DataFrame(label="Relationships", visible=False)
    graph_3d = gr.Plot(label="Knowledge Graph (3D)")
    
    with gr.Tabs():
        with gr.TabItem("Text Input"):
            gr.Markdown("## Extract Knowledge Graph from Text")
            with gr.Row():
                api_key = gr.Textbox(label="Google API Key", type="password")
            
            text_input = gr.Textbox(label="Enter text to analyze", lines=8)
            extract_btn = gr.Button("Extract Entities & Relationships")
            
            extract_output = gr.Textbox(label="Extraction Result")
            with gr.Row():
                extracted_nodes = gr.DataFrame(label="Extracted Nodes")
                extracted_relationships = gr.DataFrame(label="Extracted Relationships")
            
            extract_graph = gr.Plot(label="Knowledge Graph Visualization")
            
            extract_btn.click(
                fn=extract_knowledge_graph,
                inputs=[text_input, api_key],
                outputs=[extract_output, extracted_nodes, extracted_relationships, extract_graph]
            )
        with gr.TabItem("Manual Entry"):
            gr.Markdown("## Manual Entry")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Add Node")
                    node_id = gr.Textbox(label="Node ID")
                    node_label = gr.Textbox(label="Node Label")
                    add_node_btn = gr.Button("Add Node")
                    add_node_output = gr.Textbox(label="Result")
                    nodes_display = gr.DataFrame(label="Current Nodes")
        
                with gr.Column():
                    gr.Markdown("### Add Relationship")
                    source_node = gr.Dropdown(label="Source Node", choices=get_node_ids())
                    target_node = gr.Dropdown(label="Target Node", choices=get_node_ids())
                    rel_type = gr.Textbox(label="Relationship Type")
                    add_rel_btn = gr.Button("Add Relationship")
                    add_rel_output = gr.Textbox(label="Result")
                    rels_display = gr.DataFrame(label="Current Relationships")
            
            manual_graph = gr.Plot(label="Knowledge Graph Visualization")
            refresh_btn = gr.Button("Refresh Data")
    
            add_node_btn.click(
                fn=add_node,
                inputs=[node_id, node_label],
                outputs=[add_node_output, nodes_display, manual_graph]
            ).then(
                fn=update_dropdowns,
                outputs=[source_node, target_node]
            )
            
            add_rel_btn.click(
                fn=add_relationship,
                inputs=[source_node, target_node, rel_type],
                outputs=[add_rel_output, rels_display, manual_graph]
            )
            
            # Refresh button
            refresh_btn.click(
                fn=refresh_displays,
                outputs=[nodes_display, rels_display, manual_graph, source_node, target_node]
            )
        
        with gr.TabItem("Knowledge Graph"):
            gr.Markdown("## Knowledge Graph Visualization and Export")
            
            kg_graph = gr.Plot(label="Knowledge Graph (3D)")
            refresh_graph_btn = gr.Button("Refresh Visualization")
            
            gr.Markdown("### Neo4j Connection")
            with gr.Row():
                neo4j_uri = gr.Textbox(label="Neo4j URI", value="bolt://localhost:7687")
                neo4j_user = gr.Textbox(label="Username", value="neo4j")
                neo4j_password = gr.Textbox(label="Password", type="password")
            
            connect_btn = gr.Button("Connect to Neo4j")
            connect_output = gr.Textbox(label="Connection Result")
            
            save_neo4j_btn = gr.Button("Save to Neo4j")
            save_neo4j_output = gr.Textbox(label="Save Result")
            
            clear_btn = gr.Button("Clear All Data")
            clear_output = gr.Textbox(label="Clear Result")
            
            with gr.Accordion("Current Data", open=False):
                with gr.Row():
                    kg_nodes_display = gr.DataFrame(label="Current Nodes")
                    kg_rels_display = gr.DataFrame(label="Current Relationships")
            
            refresh_graph_btn.click(
                fn=visualize_graph_3d,
                outputs=[kg_graph]
            )
            
            refresh_tab_btn = gr.Button("Show Current Data")
            refresh_tab_btn.click(
                fn=refresh_displays,
                outputs=[kg_nodes_display, kg_rels_display, kg_graph, source_node, target_node]
            )
            
            connect_btn.click(
                fn=connect_to_neo4j,
                inputs=[neo4j_uri, neo4j_user, neo4j_password],
                outputs=[connect_output]
            )
            
            save_neo4j_btn.click(
                fn=save_to_neo4j,
                outputs=[save_neo4j_output]
            )
            
            
            clear_btn.click(
                fn=clear_data,
                outputs=[clear_output, kg_nodes_display, kg_rels_display, kg_graph]
            ).then(
                fn=update_dropdowns,
                outputs=[source_node, target_node]
            )
    
    with gr.Accordion("How to Use This App", open=False):
        gr.Markdown("""
        ## How to Use the 3D Knowledge Graph Builder
        
        ### Text Input Tab
        1. Enter your Google API key
        2. Paste text in the text area
        3. Click "Extract Entities & Relationships" to use AI to identify nodes and relationships
        
        ### Manual Entry Tab
        1. Add nodes by providing an ID and label
        2. Add relationships between nodes by selecting source and target nodes and specifying relationship type
        3. Use the "Refresh Data" button to update the visualization and dropdowns
        
        ### Knowledge Graph Tab
        1. View the 3D visualization of your knowledge graph
        2. Connect to Neo4j and save your graph to the database
        3. Click "Show Current Data" to display the current nodes and relationships
        
        ### Example Use Cases
        - Mapping relationships between people, organizations, and events
        - Creating domain knowledge models
        - Building ontologies for specific industries
        - Tracking dependencies between concepts or components
        
        ### 3D Visualization Tips
        - Click and drag to rotate the graph
        - Scroll to zoom in and out
        - Hover over nodes and edges to see details
        """)

if __name__ == "__main__":
    app.launch(debug=True)