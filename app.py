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
        charges =  atom_to_charge(atom_types).to(self.devices)
        recovery, pred_pos, pred_code = self.watermark(position=x, charges=charges, atom_types=atom_types, edge_index=edge_index, watermark=message)
        return recovery, pred_pos, pred_code

def process_data(data, batch_size=64):
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    data = np.array(data)
    num_atoms = data.shape[0]
    data = np.reshape(data, (batch_size, num_atoms//batch_size, -1))
    return data

def showmol(viewer, height=500, width=350):
    tmp_html = viewer._make_html()
    components.html(tmp_html, height=height, width=width)


def generate_molecule(mol_path, buffer=None):
    """从预生成分子文件中读取分子，或随机生成简单分子"""
    if buffer is not None:
        elements, coords, edge_index = buffer.pop()
    else:
        with open(mol_path,'rb') as f:
            mols = pickle.load(f)
        idx = np.random.randint(0, len(mols)-1)
        mol = mols[idx]
        elements = mol['x'].detach().cpu().numpy()
        coords = mol['pos'].detach().cpu().numpy()
        edge_index = mol['edge_index'].detach().cpu().numpy()
    return elements, coords, edge_index


# -------------------------
# 用户输入 -> 比特流
# -------------------------
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

def parse_xyz_text(xyz_text):
    lines = xyz_text.strip().split('\n')
    elements = []
    coords = []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        parts = line.split()
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        elements.append(elem)
        coords.append([x, y, z])
    
    return elements, np.array(coords)

# -------------------------
# RDKit 分子构建
# -------------------------
def build_mol(elements, coords):
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

def _make_global_adjacency_matrix(n_nodes, device="cpu"):
    # device = "cpu"
    row = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, -1, 1)
        .repeat(1, 1, n_nodes)
        .to(device=device)
    )
    col = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, 1, -1)
        .repeat(1, n_nodes, 1)
        .to(device=device)
    )
    full_adj = torch.concat([row, col], dim=0)
    diag_bool = torch.eye(n_nodes, dtype=torch.bool).to(device=device)
    return full_adj, diag_bool


def make_adjacency_matrix(n_nodes, full_adj, diag_bool):
    full_adj = full_adj[:, :n_nodes, :n_nodes].reshape(2, -1)
    diag_bool = diag_bool[:n_nodes, :n_nodes].reshape(-1)
    return full_adj[:, ~diag_bool]

def parse_xyz_text(xyz_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atom_one_hot = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9}
    batch_size = 64
    lines = xyz_text.strip().split('\n')
    elements = []
    coords = []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        parts = line.split()
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        elements.append([atom_one_hot[elem]])
        coords.append([x, y, z])
    
    elements = torch.tensor(elements, dtype=torch.float64, device=device)
    elements = charge_encode(elements)
    elements = elements.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elements = elements.reshape(-1, elements.shape[-1])
    elements = elements.detach().cpu().numpy().astype(np.float32)

    coords = torch.tensor(coords, dtype=torch.float64, device=device)
    coords = coords.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    for idx in range(1, batch_size):
        coords[idx] += torch.randn_like(coords[0]) * 0.1
    coords = coords.reshape(-1, coords.shape[-1])
    coords = coords.detach().cpu().numpy().astype(np.float32)

    max_n_nodes = 40
    n_nodes = elements.shape[0] // batch_size
    full_adj, diag_bool = _make_global_adjacency_matrix(max_n_nodes, device)
    edge_index_one = make_adjacency_matrix(n_nodes, full_adj, diag_bool)
    edge_index_0, edge_index_1 = [], []
    edge_index_one = edge_index_one.detach().cpu().numpy()
    idx = 0
    for idx in range(batch_size):
         edge_index_0.extend(edge_index_one[0] + idx * n_nodes)
         edge_index_1.extend(edge_index_one[1] + idx * n_nodes)

    edge_index = np.array([edge_index_0, edge_index_1], dtype=np.int64)

    return elements, coords, edge_index


mol_path = r'./generated_molecules/gen_qm9.pkl'
parameters_root ={ 
    4:r'./all_checkpoints/epoch=1819-mol_stable=0.840278-atm_stable=0.984106-validity=0.904514-recovery=0.981771.ckpt',
    6:r'./all_checkpoints/epoch=2839-mol_stable=0.876736-atm_stable=0.989281-validity=0.942708-recovery=0.958044.ckpt', 
    8:r'./all_checkpoints/epoch=1279-mol_stable=0.843750-atm_stable=0.984041-validity=0.907812-recovery=0.958398.ckpt',
    10:r'./all_checkpoints/epoch=2959-mol_stable=0.866319-atm_stable=0.987025-validity=0.927083-recovery=0.946701.ckpt',
    12:r'./all_checkpoints/epoch=1379-mol_stable=0.732639-atm_stable=0.971181-validity=0.871528-recovery=0.948495.ckpt',
    14:r'./all_checkpoints/epoch=2859-mol_stable=0.829861-atm_stable=0.982860-validity=0.923611-recovery=0.954117.ckpt',
    16:r'./all_checkpoints/epoch=2559-mol_stable=0.853125-atm_stable=0.986066-validity=0.925000-recovery=0.944922.ckpt'
    }

# -------------------------
# Streamlit 页面
# -------------------------
st.title("Demo of MolMark")

# 侧边栏参数
st.sidebar.header("Parameters Setting")
# num_atoms = st.sidebar.slider("原子数量",5,30,12,1)
sphere_radius = st.sidebar.slider("Atom Radius", 0.1, 1.0, 0.3, 0.05)
stick_radius = st.sidebar.slider("Bond Radius", 0.05, 0.5, 0.15, 0.05)
watermark_input = st.sidebar.text_input("Watermark (Text or Binary)","10101010")
max_bits = st.sidebar.slider("Maximum bits", 4, 16, 8, 2)

# 现有分子路径
mol_path = r'./generated_molecules/gen_qm9.pkl'

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate New Molecule", use_container_width=True):
        st.session_state["generate_molecule"] = True
        st.session_state["load_molecule"] = False  # 保持互斥逻辑

# 第二个按钮：加载已有分子
with col2:
    if st.button("Load Existing Molecule", use_container_width=True):
        st.session_state["load_molecule"] = True
        st.session_state["generate_molecule"] = False
  
if "generate_molecule" in st.session_state and st.session_state["generate_molecule"]:
    st.info("🧬 Step 1: Generating a new molecule...")
    st.session_state['elements'], st.session_state['coords'], st.session_state['edge_index'] = generate_molecule(mol_path)
    st.session_state["generate_pressed"] = True
    st.session_state["load_pressed"] = False
    st.session_state["generate_molecule"] = False  # 重置状态

if "load_molecule" in st.session_state and st.session_state["load_molecule"]:
    st.info("📂 Step 1: Loading existing molecule...")
    uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])
    xyz_text = st.text_area("or paste XYZ", height=250)
    if uploaded_file:
        xyz_text = uploaded_file.read().decode("utf-8")
    if xyz_text.strip() != "":
        st.session_state['elements'], st.session_state['coords'], st.session_state['edge_index'] = parse_xyz_text(xyz_text)
        st.session_state["load_pressed"] = True
        st.session_state["generate_pressed"] = False
        st.session_state["load_molecule"] = False  # 重置状态

if "generate_pressed" in st.session_state and st.session_state["generate_pressed"] or "load_pressed" in st.session_state and st.session_state["load_pressed"]:

    if "generate_pressed" in st.session_state and st.session_state['generate_pressed']:
        st.success("✅ Ready for watermark embedding for generated molecules!")
        elements = st.session_state['elements']
        coords = st.session_state['coords']
        edge_index = st.session_state['edge_index']

    if "load_pressed" in st.session_state and st.session_state['load_pressed']:
        st.success("✅ Ready for watermark embedding for loaded molecules!")
        elements = st.session_state['elements']
        coords = st.session_state['coords']
        edge_index = st.session_state['edge_index']
    

    # 用户输入转换为比特流
    bits_list = parse_bits_input(watermark_input, max_bits=max_bits)
    watermark_emb = len(bits_list)
    config = {"config_file":r"./configs/bfn4molgen_test.yaml",  
            "watermark_emb": watermark_emb,
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(**config)
    model = MolMark(config=config, device=device)

    bits_list_expand = np.array(bits_list).reshape(1, -1).repeat(config.optimization.batch_size, axis=0)

    model_param_path = parameters_root[watermark_emb]
    with open(model_param_path, 'rb') as file:
        params = torch.load(file, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.load_state_dict(params, strict=False)

    # 嵌入水印
    bitacc, wm_coords, extract_code = model.embed_watermark(elements, coords, edge_index, bits_list_expand)

    buffer_bitacc = process_data(bitacc, batch_size=config.optimization.batch_size)
    buffer_wm_coords = process_data(wm_coords, batch_size=config.optimization.batch_size)
    buffer_extract_code = process_data(extract_code, batch_size=config.optimization.batch_size)
    buffer_wm_elements = process_data(charge_decode(elements), batch_size=config.optimization.batch_size)

    buffer_coords = process_data(coords, batch_size=config.optimization.batch_size)
    buffer_elements = process_data(charge_decode(elements), batch_size=config.optimization.batch_size)

    if "generate_pressed" in st.session_state and st.session_state['generate_pressed']:
        bitacc_idx = list(enumerate(bitacc.detach().cpu().numpy()))
        idx_sorted = sorted(bitacc_idx, key=lambda x:x[1], reverse=True)
        idx = idx_sorted[0][0]
    else:
        idx = 0

    ori_coords = buffer_coords[idx]
    ori_elements = buffer_elements[idx].squeeze(1).tolist()

    watermark_coords = buffer_wm_coords[idx]
    watermark_elements = buffer_wm_elements[idx].squeeze(1).tolist()
    final_bitacc = buffer_bitacc[idx].squeeze(0)
    extracted_code = buffer_extract_code[idx].squeeze(0)


    # 构建分子
    mol_orig = build_mol(ori_elements, ori_coords * 2)
    mol_wm = build_mol(watermark_elements, watermark_coords * 2)
    mb_orig = Chem.MolToMolBlock(mol_orig)
    mb_wm = Chem.MolToMolBlock(mol_wm)

    # 左右对比展示
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Original Molecule")
        viewer_orig = py3Dmol.view(width=350,height=400)
        viewer_orig.addModel(mb_orig,"mol")
        for elem,color in element_colors.items():
            viewer_orig.setStyle({'elem':elem},{'sphere':{'radius':sphere_radius,'color':color},
                                                'stick':{'radius':stick_radius,'color':color}})
        viewer_orig.zoomTo()
        viewer_orig.setBackgroundColor('0xeeeeee')
        viewer_orig.animate({'loop':'backAndForth'})
        showmol(viewer_orig)

    with col4:
        st.subheader("Watermarked Molecule")
        viewer_wm = py3Dmol.view(width=350,height=400)
        viewer_wm.addModel(mb_wm,"mol")
        for elem,color in element_colors.items():
            viewer_wm.setStyle({'elem':elem},{'sphere':{'radius':sphere_radius,'color':color},
                                            'stick':{'radius':stick_radius,'color':color}})
        viewer_wm.zoomTo()
        viewer_wm.setBackgroundColor('0xeeeeee')
        viewer_wm.animate({'loop':'backAndForth'})
        showmol(viewer_wm)

    ori_xyz_str = f"{len(ori_elements)}\n\n"
    for elem, (x, y, z) in zip(ori_elements, ori_coords):
        ori_xyz_str += f"{elem} {x:.6f} {y:.6f} {z:.6f}\n"
    watermark_xyz_str = f"{len(watermark_elements)}\n\n"
    for elem, (x, y, z) in zip(watermark_elements, watermark_coords):
        watermark_xyz_str += f"{elem} {x:.6f} {y:.6f} {z:.6f}\n"

    col5, col6 = st.columns(2)
    with col5:
        st.download_button(
            label="💾 Download Original Molecule (.xyz)",
            data=ori_xyz_str,
            file_name="original_molecule.xyz",
            mime="chemical/x-xyz",
            key="download_original"
            )
    with col6:
        st.download_button(
            label="💾 Download Watermarked Molecule (.xyz)",
            data=watermark_xyz_str,
            file_name="watermarked_molecule.xyz",
            mime="chemical/x-xyz",
            key="download_watermarked"
            )
    

    # RMSD
    rmsd = np.sqrt(np.mean(np.sum((ori_coords-watermark_coords)**2,axis=1)))
    st.info(f"The RMSD: {rmsd:.4f} Å")

    # -------------------------
    # 水印提取与准确率率显示
    # -------------------------
    if len(bits_list) > 0:
        recovered = sum([b1 == b2 for b1, b2 in zip(bits_list, extracted_code)]) / len(bits_list)
        st.success(f"The bit accuracy: {recovered * 100:.2f}%")
        st.text(f"Original watermark:  {''.join(str(int(b)) for b in bits_list)}")
        st.text(f"Extracted watermark: {''.join(str(int(b)) for b in extracted_code)}")
    else:
        st.warning("Invalid watermark")
