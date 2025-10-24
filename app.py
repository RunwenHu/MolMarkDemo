import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
from rdkit import Chem
import numpy as np
import pickle
from core.model.watermark.encoder_decoder_test19 import WaterMarkModule
from core.config.config import Config
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
import io

# 共价半径和颜色
cov_radii = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02
}

element_colors = {
    'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green', 'P': 'orange', 'S': 'yellow', 'Cl': 'green'
}

def charge_decode(charge, atom_decoder=['H', 'C', 'N', 'O', 'F']):
    """
    charge: [n_nodes, 1]
    """
    if type(charge) is torch.Tensor:
        charge = charge.detach().cpu().numpy()
    atom_type = np.argmax(np.abs(charge), axis=-1)
    elements = [atom_decoder[i] for i in atom_type]
    return elements

def charge_encode(charge, atomic_nb=[1, 6, 7, 8, 9], remove_h=False):
    """
    atom_types: [n_nodes, 1]
    """
    atom_type_num = len(atomic_nb) - (1 if remove_h else 0)
    anchor = torch.tensor(
        [
            k for k in atomic_nb[remove_h :]
        ],
        dtype=torch.float32,
        device=charge.device,
    )
    atom_type = (charge - anchor).abs().argmin(dim=-1)
    one_hot = torch.zeros(
            [charge.shape[0], atom_type_num], dtype=torch.float32
        ).to(device=charge.device)
    one_hot[torch.arange(charge.shape[0]), atom_type] = 1
    return one_hot

def charge_to_atom(charge, atomic_nb=[1, 6, 7, 8, 9], remove_h=False):
    """
    charge: [n_nodes, 1]
    """
    atom_type_num = len(atomic_nb) - remove_h
    anchor = torch.tensor(
        [
            (2 * k - 1) / max(atomic_nb) - 1
            for k in atomic_nb[remove_h :]
        ],
        dtype=torch.float32,
        device=charge.device,
    )
    atom_type = (charge - anchor).abs().argmin(dim=-1)
    one_hot = torch.zeros(
        [charge.shape[0], atom_type_num], dtype=torch.float32
    ).to(device=charge.device)
    one_hot[torch.arange(charge.shape[0]), atom_type] = 1
    return one_hot

def atom_to_charge(atom_types, atomic_nb=[1, 6, 7, 8, 9], remove_h=False):
    """
    atom_types: [n_nodes, 5]
    """
    anchor = torch.tensor(
        [
            (2 * k - 1) / max(atomic_nb) - 1
            for k in atomic_nb[remove_h :]
        ],
        dtype=torch.float32,
        device=atom_types.device,
    )
    charge = torch.sum(anchor * atom_types, dim=-1, keepdim=True)
    
    return charge


class MolMark(pl.LightningModule):
    def __init__(self, config: Config, device):
        super().__init__()
        self.config = config
        self.watermark = WaterMarkModule(self.config, device=device)
        self.devices = device

    def embed_watermark(self, elements, coords, edge_index, bits_list):

        message = torch.Tensor(bits_list).float().to(self.devices)
        x = torch.tensor(coords).to(self.devices)
        edge_index =  torch.tensor(edge_index).to(self.devices)
        atom_types =  torch.tensor(elements).to(self.devices)
        charges = atom_to_charge(atom_types).to(self.devices)
        pred_pos = self.watermark.embed_watermark(position=x, charges=charges, atom_types=atom_types, edge_index=edge_index, watermark=message)

        return pred_pos
    
    def extract_watermark(self, elements, coords, edge_index):

        x = torch.tensor(coords).to(self.devices)
        edge_index =  torch.tensor(edge_index).to(self.devices)
        atom_types =  torch.tensor(elements).to(self.devices)
        charges = atom_to_charge(atom_types).to(self.devices)
        pred_pos = self.watermark.extract_watermark(position=x, charges=charges, atom_types=atom_types, edge_index=edge_index)

        return pred_pos
    

def process_data(data, batch_size=64):
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    data = np.array(data)
    num_atoms = data.shape[0]
    data = np.reshape(data, (batch_size, num_atoms//batch_size, -1))
    return data

def showmol(viewer, height=250, width=250):
    tmp_html = viewer._make_html()
    components.html(tmp_html, height=height, width=width)


def preprocess_data(elements, coords, edge_index, batch_size=64, device=torch.device('cpu')):

    n_nodes = elements.shape[0]

    elements = torch.tensor(elements, dtype=torch.float32, device=device)
    elements = elements.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elements = elements.reshape(-1, elements.shape[-1])

    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    coords = coords.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    coords = coords.reshape(-1, coords.shape[-1])
    
    edge_index_0, edge_index_1 = [], []
    idx = 0
    for idx in range(batch_size):
         edge_index_0.extend(edge_index[0] + idx * n_nodes)
         edge_index_1.extend(edge_index[1] + idx * n_nodes)
    edge_index = np.array([edge_index_0, edge_index_1], dtype=np.int64)

    return elements, coords, edge_index


def generate_molecule(mol_path, num_atoms=12):
    
    with open(mol_path,'rb') as f:
        mols = pickle.load(f)
    selected_mols = mols[num_atoms]
    idx = np.random.randint(0, len(selected_mols))
    mol = selected_mols[idx]
    elements = np.array(mol['x'], dtype=np.float32)
    coords = np.array(mol['pos'], dtype=np.float32)
    edge_index = np.array(mol['edge_index'], dtype=np.int64)

    return elements, coords, edge_index

def parse_bits_input(user_input, max_bits=64):
    """支持二进制字符串或文本"""
    user_input = user_input.strip()
    bits = []
    if all(c in '01' for c in user_input):
        # 直接是二进制字符串
        bits = [int(c) for c in user_input]
    else:
        # 文本转 ASCII -> 二进制
        for c in user_input:
            bits.extend([int(b) for b in format(ord(c),'08b')])
    if len(bits) % 2 != 0:
        bits.append(0)
    return bits[:max_bits]


# -------------------------
# RDKit 分子构建
# -------------------------
def build_mol(elements, coords):
    elements = charge_decode(elements)
    mol = Chem.RWMol()
    for elem in elements:
        mol.AddAtom(Chem.Atom(elem))
    N = len(elements)
    for i in range(N):
        for j in range(i+1, N):
            r1 = cov_radii.get(elements[i], 0.7)
            r2 = cov_radii.get(elements[j], 0.7)
            if np.linalg.norm(coords[i]-coords[j]) <= r1 + r2 + 0.4:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    conf = Chem.Conformer(N)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def parse_xyz_text(ori_data):

    elements = ori_data.elements
    coords = ori_data.coords
    edge_index = ori_data.edge_index

    return elements, coords, edge_index

def embed_watermark_with_model(config, elements, coords, edge_index, bits_list):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MolMark(config=config, device=device)
    bits_list_expand = np.array(bits_list).reshape(1, -1).repeat(config.optimization.batch_size, axis=0)
    model_param_path = parameters_root[config.watermark_emb]
    with open(model_param_path, 'rb') as file:
        params = torch.load(file, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.load_state_dict(params['state_dict'], strict=False)
        # model.load_state_dict(params, strict=False)

    # 嵌入水印
    wm_coords = model.embed_watermark(elements, coords, edge_index, bits_list_expand)
    wm_coords = wm_coords.detach().cpu().numpy()

    return wm_coords, elements

def extract_watermark_with_model(config, elements, coords, edge_index):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = MolMark(config=config, device=device)
    model_param_path = parameters_root[config.watermark_emb]
    with open(model_param_path, 'rb') as file:
        params = torch.load(file, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.load_state_dict(params['state_dict'], strict=False)
        # model.load_state_dict(params, strict=False)

    # 提取水印
    extract_code = model.extract_watermark(elements, coords, edge_index)
    extract_code = extract_code.detach().cpu().numpy()

    return extract_code


def styled_title(text, size=18, color="#333", align="left", weight="600", underline=False):
    """
    输出自定义样式标题文字
    """
    text_decoration = "underline" if underline else "none"
    html = f"""
    <p style="
        font-size:{size}px;
        color:{color};
        font-weight:{weight};
        text-align:{align};
        text-decoration:{text_decoration};
        margin:0 0 8px 0;
    ">
        {text}
    </p>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_multiple_molecules(elements, coords, edge_index, element_colors, sphere_radius, stick_radius, num_molecules=3, batch_size=64, kind="Original"):

    """展示多个分子（默认最多展示3个），并自动添加空间偏移防止重叠"""

    # 下载 pkl 数据
    ori_data = Data(
        elements=elements,
        coords=coords,
        edge_index=edge_index,
    )
    buffer = io.BytesIO()
    pickle.dump(ori_data, buffer)
    buffer.seek(0)

    
    st.download_button(
        label=f"💾 Download {kind} Molecules (.pkl)",
        data=buffer,
        file_name=f"{kind}_Molecules.pkl",
        mime="application/octet-stream",
        key=f"download_{kind}_molecules"
    )

    elements = np.reshape(elements, (batch_size, 1, -1, elements.shape[-1]))
    coords = np.reshape(coords, (batch_size, 1, -1, coords.shape[-1]))
    num_total = elements.shape[0]
    num_show = min(num_molecules, num_total)
    cols = st.columns(num_show)

    # 空间偏移向量
    offsets = np.linspace(-6, 6, num_show)  # 让分子在3D中分开显示
    offsets = np.stack([offsets, np.zeros(num_show), np.zeros(num_show)], axis=1)

    per_column = 3

    for row_start in range(0, num_show, per_column):
        row_end = min(row_start + per_column, num_show)
        cols = st.columns(row_end - row_start)
        for idx, i in enumerate(range(row_start, row_end)):
            with cols[idx]:
                elements_i = elements[i, 0, :, :]
                coords_i = coords[i, 0, :, :].copy()
                coords_i = coords_i + offsets[i]

                mol_i = build_mol(elements_i, np.array(coords_i) * 2)
                mb_i = Chem.MolToMolBlock(mol_i)

                styled_title(f"{kind} Molecule {i+1}", size=18, color="#444", align="left")
                # 渲染
                width = height = 200
                viewer = py3Dmol.view(width=width, height=height)
                viewer.addModel(mb_i, "mol")
                for elem, color in element_colors.items():
                    viewer.setStyle({'elem': elem}, {
                        'sphere': {'radius': sphere_radius, 'color': color},
                        'stick': {'radius': stick_radius, 'color': color}
                    })
                viewer.zoomTo()
                viewer.setBackgroundColor('0xeeeeee')
                viewer.animate({'loop': 'backAndForth'})
                showmol(viewer, width=width, height=height)

# 现有分子路径
mol_path = r'./generated_molecules/gen_qm9_new.pkl'
parameters_root ={ 
    4:r'./all_checkpoints/epoch=19-mol_stable=0.781250-atm_stable=0.971690-validity=0.944444-recovery=0.996962.ckpt',
    6:r'./all_checkpoints/epoch=19-mol_stable=0.865234-atm_stable=0.985585-validity=0.957031-recovery=0.994141.ckpt',  
    8:r'./all_checkpoints/epoch=19-mol_stable=0.644097-atm_stable=0.926154-validity=0.875000-recovery=0.999132.ckpt',
    10:r'./all_checkpoints/epoch=19-mol_stable=0.845486-atm_stable=0.981175-validity=0.949653-recovery=0.945486.ckpt',
    12:r'./all_checkpoints/epoch=19-mol_stable=0.907986-atm_stable=0.991429-validity=0.979167-recovery=0.982928.ckpt',
    14:r'./all_checkpoints/epoch=19-mol_stable=0.909722-atm_stable=0.991443-validity=0.954861-recovery=0.984623.ckpt',
    16:r'./all_checkpoints/epoch=19-mol_stable=0.602431-atm_stable=0.915827-validity=0.875000-recovery=0.971354.ckpt',
    }

# -------------------------
# Streamlit 页面
# -------------------------
def setup_app_style():
    """全局页面样式与Sidebar布局设置（带参数重置功能）"""

    # ===== 🎨 全局 CSS 样式 =====
    st.markdown("""
    <style>
    .main { background-color: #f9fafc; }
    h1 {
        background: linear-gradient(90deg, #0072ff, #8000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 15px;
    }
    [data-testid="stSidebar"] {
        background-color: #f3f6fa;
        padding: 1.5rem 1rem 1rem 1rem;
        border-right: 2px solid #e0e0e0;
    }
    [data-testid="stSidebar"] h2 {
        color: #1565C0;
        font-size: 18px;
        font-weight: 500;
        text-align: left;
    }
    div[data-testid="stSidebar"] > div { margin-bottom: 10px; }
    .stSlider label, .stNumberInput label, .stTextInput label {
        font-weight: 600;
        color: #0d47a1;
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #555555, #bbbbbb);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #1976d2, #00bcd4);
    }
    .molecule-title {
        font-size: 22px;
        font-weight: 700;
        color: #283593;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===== 🌟 页面标题 =====
    st.markdown("<h1>Demo of MolMark</h1>", unsafe_allow_html=True)

    # ===== 🧭 Sidebar 参数 =====
    st.sidebar.header("⚙️ Parameters Setting")

    # --- 默认参数 ---
    default_values = {
        "num_molecules": 3,
        "num_atoms": 20,
        "sphere_radius": 0.3,
        "stick_radius": 0.15,
        "max_bits": 8,
        "watermark_input": "1010101010101010",
    }

    # --- 初始化 session_state ---
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- 参数输入组件 ---
    col_s1, col_s2 = st.sidebar.columns(2)

    # Molecule 数量下拉
    num_molecule_options = list(range(1, 65))  # 可选 1~64
    st.session_state.num_molecules = col_s1.selectbox(
        "Molecule",
        options=num_molecule_options,
        index=st.session_state.num_molecules - 1  # 默认选中当前值
    )

    # Atom 数量下拉
    num_atom_options = list(range(10, 28))  # 可选 10~27
    st.session_state.num_atoms = col_s2.selectbox(
        "Atom",
        options=num_atom_options,
        index=st.session_state.num_atoms - 10  # 默认选中当前值
    )

    col_s3, col_s4 = st.sidebar.columns(2)

    # Atom Radius 下拉
    atom_radius_options = [round(0.1 + 0.05*i, 2) for i in range(19)]  # 0.1~1.0
    st.session_state.sphere_radius = col_s3.selectbox(
        "Atom Radius",
        options=atom_radius_options,
        index=atom_radius_options.index(st.session_state.sphere_radius)
    )

    # Bond Radius 下拉
    bond_radius_options = [round(0.05 + 0.05*i, 2) for i in range(10)]  # 0.05~0.5
    st.session_state.stick_radius = col_s4.selectbox(
        "Bond Radius",
        options=bond_radius_options,
        index=bond_radius_options.index(st.session_state.stick_radius)
    )

    # Maximum bits 下拉
    max_bits_options = [4, 6, 8, 10, 12, 14, 16]
    st.session_state.max_bits = st.sidebar.selectbox(
        "Maximum bits",
        options=max_bits_options,
        index=max_bits_options.index(st.session_state.max_bits)
    )
    st.session_state.watermark_input = st.sidebar.text_input(
        "Watermark (Text or Binary)", st.session_state.watermark_input
    )

    # --- 🧹 参数重置按钮 ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Parameters"):
        for key, value in default_values.items():
            st.session_state[key] = value


    # 返回参数
    return (
        st.session_state.num_molecules,
        st.session_state.num_atoms,
        st.session_state.sphere_radius,
        st.session_state.stick_radius,
        st.session_state.max_bits,
        st.session_state.watermark_input,
    )

num_molecules, num_atoms, sphere_radius, stick_radius, max_bits, watermark_input = setup_app_style()

bits_list = parse_bits_input(watermark_input, max_bits=max_bits)
watermark_emb = len(bits_list)
config = {"config_file":r"./configs/bfn4molgen_test.yaml",  
        "watermark_emb": watermark_emb, }
config = Config(**config)


st.header("Watermark Operations")
col1, col2 = st.columns(2)

with col1:
    if st.button("🧬 Generate New Molecule", use_container_width=True):
        st.session_state["generate_molecule"] = True
        st.session_state["load_molecule"] = False  # 保持互斥逻辑


# 第二个按钮：加载已有分子
with col2:
    if st.button("📂 Load Existing Molecule", use_container_width=True):
        st.session_state["load_molecule"] = True
        st.session_state["generate_molecule"] = False


col_embed, col_extract = st.columns(2)
with col_embed:
    if st.button("🧩 Embed Watermark", use_container_width=True):
        st.session_state["embed_watermark"] = True

with col_extract:
    if st.button("🔍 Extract Watermark", use_container_width=True):
        st.session_state["extract_watermark"] = True
  

if "generate_molecule" in st.session_state and st.session_state["generate_molecule"]:
    # st.info("🧬 Step 1: Generating a new molecule...")

    output = generate_molecule(mol_path, num_atoms)
    st.session_state['ori_elements'], st.session_state['ori_coords'], st.session_state['ori_edge_index'] = output

    st.session_state["generate_pressed"] = True
    st.session_state["load_pressed"] = False
    st.session_state["generate_molecule"] = False  # 重置状态
    st.session_state["embed_watermark"] = False  # 重置状态
    st.session_state["extract_watermark"] = False  # 重置状态

if "load_molecule" in st.session_state and st.session_state["load_molecule"]:
    uploaded_file = st.file_uploader("Upload PKL file", type=["pkl"])
    st.session_state["generate_pressed"] = False
    st.session_state["embed_watermark"] = False  # 重置状态
    st.session_state["extract_watermark"] = False  # 重置状态
    if uploaded_file is not None:
        ori_data = pickle.load(uploaded_file)
        output = parse_xyz_text(ori_data)
        st.session_state['ori_elements'], st.session_state['ori_coords'], st.session_state['ori_edge_index'] = output
        st.session_state["load_pressed"] = True
        st.session_state["load_molecule"] = False  # 重置状态
        
        

if "generate_pressed" in st.session_state and st.session_state["generate_pressed"] or "load_pressed" in st.session_state and st.session_state["load_pressed"]:

    if "generate_pressed" in st.session_state and st.session_state['generate_pressed'] or "load_pressed" in st.session_state and st.session_state['load_pressed']:
        ori_elements = st.session_state['ori_elements']
        ori_coords = st.session_state['ori_coords']
        ori_edge_index = st.session_state['ori_edge_index']
        st.session_state['original_molecule'] = True


    if "original_molecule" in st.session_state and st.session_state["original_molecule"]:

        display_multiple_molecules(
            ori_elements,
            ori_coords,
            ori_edge_index,
            element_colors,
            sphere_radius,
            stick_radius,
            num_molecules,
            kind='Original'
        )


    if "embed_watermark" in st.session_state and st.session_state["embed_watermark"]:
       
        watermark_coords, watermark_elements = embed_watermark_with_model(config, ori_elements, ori_coords, ori_edge_index, bits_list)
        st.session_state["watermark_coords"] = watermark_coords
        st.session_state["watermark_elements"] = watermark_elements
        st.session_state["watermark_edge_index"] = ori_edge_index
        st.session_state["watermarked_molecule"] = True

        display_multiple_molecules(
            watermark_elements,
            watermark_coords,
            ori_edge_index,
            element_colors,
            sphere_radius,
            stick_radius,
            num_molecules,
            kind='Watermarked'
        )

    if "extract_watermark" in st.session_state and st.session_state["extract_watermark"]:
        
        if "embed_watermark" in st.session_state and st.session_state["embed_watermark"]:
            # 使用嵌入水印的分子进行提取
            extract_elements = st.session_state["watermark_elements"]
            extract_coords = st.session_state["watermark_coords"]
            extract_edge_index = st.session_state["watermark_edge_index"]
            extracted_code = extract_watermark_with_model(config, extract_elements, extract_coords, extract_edge_index)
            extracted_code = np.reshape(extracted_code, (config.optimization.batch_size, -1))

            rmsd = np.sqrt(np.mean(np.sum((ori_coords - watermark_coords) ** 2, axis=1)))
            st.info(f"The RMSD: {rmsd:.4f} Å")
            if len(bits_list) > 0:
                num_correct = []
                total = 0
                for idx in range(config.optimization.batch_size):
                    temp_code = extracted_code[idx, :]
                    temp_val = sum([b1 == b2 for b1, b2 in zip(bits_list, temp_code)])
                    num_correct.append(temp_val)
                    total += temp_val
                recovered = total / (len(bits_list) * config.optimization.batch_size)

                st.success(f"The average bit accuracy: {recovered * 100:.2f}%")
                st.text(f"Original watermark:  {''.join(str(int(b)) for b in bits_list)}")
                for i in range(num_molecules):
                    bitacc = num_correct[i] / len(bits_list)
                    st.text(f"For molecule {i}: ACC is {bitacc * 100:.2f}, extracted watermark is {''.join(str(int(b)) for b in extracted_code[i, :])}")
            else:
                st.warning("Invalid watermark")

        elif "load_pressed" in st.session_state and st.session_state['load_pressed']:

            extract_elements = st.session_state['ori_elements']
            extract_coords = st.session_state['ori_coords']
            extract_edge_index = st.session_state['ori_edge_index']
            extracted_code = extract_watermark_with_model(config, extract_elements, extract_coords, extract_edge_index)

            if len(bits_list) > 0:
                num_correct = []
                total = 0
                for idx in range(config.optimization.batch_size):
                    temp_code = extracted_code[idx, :]
                    temp_val = sum([b1 == b2 for b1, b2 in zip(bits_list, temp_code)])
                    num_correct.append(temp_val)
                    total += temp_val
                recovered = total / (len(bits_list) * config.optimization.batch_size)

                st.success(f"The average bit accuracy: {recovered * 100:.2f}%")
                st.text(f"Original watermark:  {''.join(str(int(b)) for b in bits_list)}")   
                for i in range(num_molecules):
                    bitacc = num_correct[i] / len(bits_list)
                    st.text(f"For molecule {i}: ACC is {bitacc * 100:.2f}, extracted watermark is {''.join(str(int(b)) for b in extracted_code[i, :])}")
            else:
                st.warning("Invalid watermark")
        else:
            st.warning("Please embed watermark first or use the watermarked molecule for extraction.")
