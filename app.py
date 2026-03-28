import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
import io
import pandas as pd
from datetime import datetime
import os
import warnings
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from supabase import create_client

warnings.filterwarnings('ignore')

# ── Connexion Supabase ────────────────────────────────────────────
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="MindCheck", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# ── Session state ─────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]
if "langue" not in st.session_state:
    st.session_state.langue = "FR"
if "model" not in st.session_state:
    st.session_state.model = None

# ── Style ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-title {
        text-align: center; font-size: 4em; font-weight: bold;
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; font-size: 1.2em; color: rgba(255,255,255,0.9); margin-bottom: 30px; }
    .modern-card {
        background: rgba(255,255,255,0.95); border-radius: 20px; padding: 20px;
        margin: 10px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .result-card { border-radius: 20px; padding: 20px; text-align: center; color: white; margin: 10px 0; }
    .result-good { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .result-bad  { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .stButton > button {
        width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 50px; padding: 12px 30px;
        font-size: 1.1em; font-weight: bold;
    }
    .stDownloadButton > button {
        width: 100%; background: #27AE60; color: white;
        border: none; border-radius: 50px; padding: 12px 30px; font-size: 1.1em;
    }
    .install-steps {
        background: rgba(255,255,255,0.95); border-radius: 20px;
        padding: 20px; margin: 20px 0; color: #333;
    }
    .footer { text-align: center; color: rgba(255,255,255,0.7); font-size: 0.8em; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# ── Traductions ───────────────────────────────────────────────────
LANG = {
    "FR": {
        "titre": "🧠 MindCheck",
        "sous_titre": "Ton bien-être mental en 30 secondes ⚡",
        "infos": "👤 Qui es-tu ?",
        "travail": "💼 Ton quotidien",
        "sante": "🧘 Ta santé mentale",
        "age": "🎂 Âge",
        "genre": "⚧ Genre",
        "genre_opts": ["👨 Homme", "👩 Femme", "🌈 Autre"],
        "remote": "🏠 Télétravail ?",
        "benefits": "💊 Avantages santé ?",
        "seek": "🤝 Soutien employeur ?",
        "anon": "🔒 Anonymat ?",
        "family": "👨‍👩‍👧 Antécédents familiaux ?",
        "work_impact": "⚡ Impact au travail ?",
        "oui_non": ["✅ Oui", "❌ Non"],
        "oui_non_nsp": ["✅ Oui", "❌ Non", "🤔 Je ne sais pas"],
        "impact_opts": ["🔥 Souvent", "😐 Parfois", "😌 Rarement", "😊 Jamais"],
        "bouton": "✨ Analyser mon bien-être ✨",
        "resultat": "📊 Ton résultat",
        "risque": "⚠️ Niveau de vigilance",
        "bienetre": "💚 Niveau de bien-être",
        "reco_titre": "💡 Recommandations",
        "reco": ["🗣️ Parle à un professionnel", "📞 Appelle une ligne d'écoute", "🧘 Prends 5 minutes pour toi", "💬 Parle à un proche"],
        "conseils_titre": "🌟 Conseils pour toi",
        "conseils": ["😴 Dors 7-8h par nuit", "🏃 Bouge ton corps", "🥗 Mange équilibré", "👥 Garde du lien social"],
        "radar": "🎯 Ton profil en un coup d'œil",
        "detail": "📋 Détail des facteurs",
        "pourquoi": "🤖 Pourquoi ce résultat ?",
        "pdf_titre": "📄 Ton rapport personnalisé",
        "pdf_btn": "⬇️ Télécharger mon rapport PDF",
        "histo_titre": "📜 Ton historique",
        "histo_vide": "Aucune analyse pour le moment",
        "evolution": "📈 Ton évolution",
        "effacer": "🗑️ Effacer mon historique",
        "effacer_ok": "Historique effacé !",
        "footer": "MindCheck © 2025 — Prends soin de toi 💜",
        "analyse_info": "💡 Principalement influencé par",
        "et": "et",
        "cat_radar": ["Famille","Travail","Soutien","Avantages","Anonymat","Télétravail"],
        "cat_barres": ["👨‍👩‍👧 Famille","⚡ Travail","🤝 Soutien","💊 Avantages","🔒 Anonymat","🏠 Télétravail"],
        "feature_names": ["Âge","Genre","Famille","Travail","Télétravail","Avantages","Soutien","Anonymat"],
        "importance_label": "Impact (%)",
        "importance_titre": "Ce qui influence ton score",
        "score_label": "Score (%)",
        "analyses_label": "Analyses",
        "entrainer": "🎮 Lancer l'analyse intelligente",
        "meilleur_modele": "🏆 Modèle optimal",
        "entrainement_success": "✅ Analyse prête !",
        "welcome": "✨ Bienvenue sur MindCheck ✨",
        "welcome_text": "Découvre en quelques secondes comment va ton bien-être mental",
        "questions": "📝 Quelques questions",
    },
    "EN": {
        "titre": "🧠 MindCheck",
        "sous_titre": "Your mental wellness in 30 seconds ⚡",
        "infos": "👤 Who are you?",
        "travail": "💼 Your daily life",
        "sante": "🧘 Your mental health",
        "age": "🎂 Age",
        "genre": "⚧ Gender",
        "genre_opts": ["👨 Male", "👩 Female", "🌈 Other"],
        "remote": "🏠 Remote work?",
        "benefits": "💊 Health benefits?",
        "seek": "🤝 Employer support?",
        "anon": "🔒 Anonymity?",
        "family": "👨‍👩‍👧 Family history?",
        "work_impact": "⚡ Work impact?",
        "oui_non": ["✅ Yes", "❌ No"],
        "oui_non_nsp": ["✅ Yes", "❌ No", "🤔 Don't know"],
        "impact_opts": ["🔥 Often", "😐 Sometimes", "😌 Rarely", "😊 Never"],
        "bouton": "✨ Analyze my wellness ✨",
        "resultat": "📊 Your result",
        "risque": "⚠️ Attention level",
        "bienetre": "💚 Wellness level",
        "reco_titre": "💡 Recommendations",
        "reco": ["🗣️ Talk to a professional", "📞 Call a helpline", "🧘 Take 5 min for yourself", "💬 Talk to someone"],
        "conseils_titre": "🌟 Tips for you",
        "conseils": ["😴 Sleep 7-8h", "🏃 Move your body", "🥗 Eat balanced", "👥 Stay connected"],
        "radar": "🎯 Your profile at a glance",
        "detail": "📋 Factor details",
        "pourquoi": "🤖 Why this result?",
        "pdf_titre": "📄 Your personalized report",
        "pdf_btn": "⬇️ Download PDF report",
        "histo_titre": "📜 Your history",
        "histo_vide": "No analyses yet",
        "evolution": "📈 Your evolution",
        "effacer": "🗑️ Clear my history",
        "effacer_ok": "History cleared!",
        "footer": "MindCheck © 2025 — Take care of yourself 💜",
        "analyse_info": "💡 Mainly influenced by",
        "et": "and",
        "cat_radar": ["Family","Work","Support","Benefits","Anonymity","Remote"],
        "cat_barres": ["👨‍👩‍👧 Family","⚡ Work","🤝 Support","💊 Benefits","🔒 Anonymity","🏠 Remote"],
        "feature_names": ["Age","Gender","Family","Work","Remote","Benefits","Support","Anonymity"],
        "importance_label": "Impact (%)",
        "importance_titre": "What influences your score",
        "score_label": "Score (%)",
        "analyses_label": "Analyses",
        "entrainer": "🎮 Launch smart analysis",
        "meilleur_modele": "🏆 Optimal model",
        "entrainement_success": "✅ Analysis ready!",
        "welcome": "✨ Welcome to MindCheck ✨",
        "welcome_text": "Discover your mental wellness in seconds",
        "questions": "📝 A few questions",
    }
}

# ── Fonctions utilitaires ─────────────────────────────────────────
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    age      = np.random.randint(18, 65, n_samples)
    gender   = np.random.choice([0,1,2], n_samples)
    family   = np.random.choice([0,1], n_samples, p=[0.7,0.3])
    work     = np.random.choice([0,1,2,3], n_samples)
    remote   = np.random.choice([0,1], n_samples)
    benefits = np.random.choice([0,1,2], n_samples)
    seek     = np.random.choice([0,1,2], n_samples)
    anon     = np.random.choice([0,1,2], n_samples)
    risk = (family*0.3 + (work/3)*0.25 + (1-seek/2)*0.2 +
            (1-benefits/2)*0.15 + (1-anon/2)*0.1 + remote*0.05)
    risk += np.random.normal(0, 0.08, n_samples)
    risk = np.clip(risk, 0, 1)
    target = (risk > 0.5).astype(int)
    X = np.column_stack([age, gender, family, work, remote, benefits, seek, anon])
    return X, target

def train_models():
    with st.spinner("🎮 Analyse en cours..."):
        X, y = generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
        }
        best_model, best_f1, best_name = None, 0, ""
        for name, m in models.items():
            m.fit(X_train, y_train)
            f1 = f1_score(y_test, m.predict(X_test))
            if f1 > best_f1:
                best_f1, best_model, best_name = f1, m, name
        return best_model, best_name

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

def sauvegarder(age, gender, family, work, score_risque, resultat, langue):
    try:
        supabase.table("historique").insert({
            "user_id":      st.session_state.user_id,
            "date":         datetime.now().strftime("%d/%m/%Y %H:%M"),
            "langue":       langue,
            "age":          age,
            "genre":        gender,
            "antecedents":  family,
            "impact_travail": work,
            "score_risque": f"{score_risque}%",
            "resultat":     "Risque" if resultat == 1 else "Bien-être"
        }).execute()
    except Exception as e:
        st.warning(f"Historique non sauvegardé : {e}")

def generer_pdf(age, gender, family, work, remote, benefits, seek, anon,
                score_risque, score_bienetre, resultat, valeurs, importances, feature_names, indices):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    contenu = []
    ts = ParagraphStyle('t', parent=styles['Title'], fontSize=22,
                        textColor=colors.HexColor("#667eea"), spaceAfter=6)
    ss = ParagraphStyle('s', parent=styles['Normal'], fontSize=11,
                        textColor=colors.gray, spaceAfter=20)
    hs = ParagraphStyle('h', parent=styles['Heading2'], fontSize=13,
                        textColor=colors.HexColor("#2C3E50"), spaceBefore=14, spaceAfter=8)
    bs = ParagraphStyle('b', parent=styles['Normal'], fontSize=10,
                        leading=16, textColor=colors.HexColor("#333333"))
    contenu.append(Paragraph("MindCheck — Mental Health Report", ts))
    contenu.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ss))
    contenu.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#667eea")))
    contenu.append(Spacer(1, 0.4*cm))
    cl  = "#E74C3C" if resultat == 1 else "#2ECC71"
    lbl = f"Risk score: {score_risque}%" if resultat == 1 else f"Well-being score: {score_bienetre}%"
    contenu.append(Paragraph("Result", hs))
    contenu.append(Paragraph(lbl, ParagraphStyle('r', parent=styles['Normal'],
                             fontSize=14, textColor=colors.HexColor(cl), fontName="Helvetica-Bold")))
    contenu.append(Spacer(1, 0.3*cm))
    contenu.append(Paragraph("Input information", hs))
    data = [["Criterion","Value"],["Age",str(age)],["Gender",gender],
            ["Family history",family],["Work impact",work],["Remote work",remote],
            ["Health benefits",benefits],["Employer support",seek],["Anonymity",anon]]
    t = Table(data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#667eea")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor("#F8F9FA"),colors.white]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#DDDDDD")),
        ('PADDING',(0,0),(-1,-1),8),
    ]))
    contenu.append(t)
    contenu.append(Spacer(1, 0.4*cm))
    f1,f2 = feature_names[indices[0]], feature_names[indices[1]]
    p1,p2 = importances[indices[0]]*100, importances[indices[1]]*100
    contenu.append(Paragraph("Key AI factors", hs))
    contenu.append(Paragraph(f"Mainly influenced by <b>{f1}</b> ({p1:.1f}%) and <b>{f2}</b> ({p2:.1f}%).", bs))
    contenu.append(Spacer(1, 0.4*cm))
    contenu.append(Paragraph("Recommendations", hs))
    recs = ["Consult a mental health professional.", "Take time for yourself daily.",
            "Talk to someone you trust."] if resultat == 1 else \
           ["Maintain good sleep habits.", "Exercise regularly.", "Stay socially connected."]
    for r in recs:
        contenu.append(Paragraph(f"• {r}", bs))
    contenu.append(Spacer(1, 0.6*cm))
    contenu.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    contenu.append(Paragraph("MindCheck © 2025 — This tool does not replace professional medical advice.",
                             ParagraphStyle('f', parent=styles['Normal'],
                                            fontSize=8, textColor=colors.gray, alignment=1)))
    doc.build(contenu)
    buffer.seek(0)
    return buffer

def add_install_qr():
    import qrcode
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=4)
    qr.add_data("https://mindcheck.streamlit.app")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#667eea", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    st.markdown("""
    <div style='text-align:center; background:rgba(255,255,255,0.1);
    border-radius:20px; padding:20px; margin:20px 0;'>
        <h4 style='color:white'>📱 Installe MindCheck sur ton téléphone</h4>
        <p style='color:rgba(255,255,255,0.8); font-size:0.85em;'>
            🔹 Scanne ce QR code<br>🔹 Puis suis les instructions
        </p>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.image(buf, width=200)
    st.markdown("""
    <div class='install-steps'>
        <h4>📲 Comment installer ?</h4>
        <div style='display:flex; gap:20px; flex-wrap:wrap;'>
            <div style='flex:1; min-width:200px;'>
                <strong>📱 Android :</strong>
                <ol><li>Ouvre Chrome</li><li>3 points ⋮ → Ajouter à l'écran d'accueil</li></ol>
            </div>
            <div style='flex:1; min-width:200px;'>
                <strong>🍎 iPhone :</strong>
                <ol><li>Ouvre Safari</li><li>Partager 📤 → Sur l'écran d'accueil</li></ol>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Interface ─────────────────────────────────────────────────────
col_title, col_lang, col_toggle = st.columns([3, 1, 1])
with col_lang:
    langue = st.selectbox("🌍", ["FR", "EN"],
                          index=0 if st.session_state.langue == "FR" else 1,
                          label_visibility="collapsed")
    st.session_state.langue = langue
with col_toggle:
    st.write("")

T = LANG[langue]

st.markdown(f'<h1 class="main-title">{T["titre"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{T["sous_titre"]}</p>', unsafe_allow_html=True)

# ── Chargement/entraînement modèle ────────────────────────────────
model = load_model()

if model is None and st.session_state.model is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""<div class="modern-card" style="text-align:center;">
            <h2>{T['welcome']}</h2><p>{T['welcome_text']}</p></div>""", unsafe_allow_html=True)
        if st.button(T["entrainer"], use_container_width=True):
            best_model, best_name = train_models()
            st.session_state.model = best_model
            with open("model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            st.success(f"✅ {T['entrainement_success']} — {T['meilleur_modele']}: {best_name}")
            st.rerun()

# ── Questionnaire ─────────────────────────────────────────────────
model_to_use = st.session_state.model if st.session_state.model is not None else model

if model_to_use:
    st.markdown(f'<div class="modern-card"><h3>{T["questions"]}</h3>', unsafe_allow_html=True)

    st.markdown(f"### {T['infos']}")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(T["age"], 18, 65, 25)
    with col2:
        gender = st.selectbox(T["genre"], T["genre_opts"])

    st.markdown(f"### {T['travail']}")
    col3, col4 = st.columns(2)
    with col3:
        remote   = st.selectbox(T["remote"],   T["oui_non"])
        benefits = st.selectbox(T["benefits"], T["oui_non_nsp"])
    with col4:
        seek = st.selectbox(T["seek"], T["oui_non_nsp"])
        anon = st.selectbox(T["anon"], T["oui_non_nsp"])

    st.markdown(f"### {T['sante']}")
    col5, col6 = st.columns(2)
    with col5:
        family = st.selectbox(T["family"],      T["oui_non"])
    with col6:
        work   = st.selectbox(T["work_impact"], T["impact_opts"])

    gender_enc   = T["genre_opts"].index(gender)
    family_enc   = T["oui_non"].index(family)
    work_enc     = T["impact_opts"].index(work)
    remote_enc   = T["oui_non"].index(remote)
    benefits_enc = T["oui_non_nsp"].index(benefits)
    seek_enc     = T["oui_non_nsp"].index(seek)
    anon_enc     = T["oui_non_nsp"].index(anon)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(T["bouton"], use_container_width=True):
            data_input = [[age, gender_enc, family_enc, work_enc, remote_enc, benefits_enc, seek_enc, anon_enc]]
            resultat       = model_to_use.predict(data_input)[0]
            proba          = model_to_use.predict_proba(data_input)[0]
            score_risque   = int(proba[1] * 100)
            score_bienetre = 100 - score_risque

            sauvegarder(age, gender, family, work, score_risque, resultat, langue)

            if resultat == 1:
                st.markdown(f"""<div class="result-card result-bad">
                    <h2>{T['risque']}</h2><h1>{score_risque}%</h1>
                    <p>Prends soin de toi 💜</p></div>""", unsafe_allow_html=True)
                st.markdown(f"### {T['reco_titre']}")
                for r in T["reco"]:
                    st.markdown(f"- {r}")
            else:
                st.markdown(f"""<div class="result-card result-good">
                    <h2>{T['bienetre']}</h2><h1>{score_bienetre}%</h1>
                    <p>Continue comme ça ! 🌟</p></div>""", unsafe_allow_html=True)
                st.markdown(f"### {T['conseils_titre']}")
                for c in T["conseils"]:
                    st.markdown(f"- {c}")

            st.progress(score_risque / 100)

            # Radar avec matplotlib (version corrigée et fiable)
            st.markdown(f"### {T['radar']}")
            valeurs = [family_enc, work_enc/3, 1-seek_enc/2, 1-benefits_enc/2, 1-anon_enc/2, remote_enc]
            categories = T["cat_radar"]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('rgba(255,255,255,0.1)')
            
            angles = [n / 6 * 2 * np.pi for n in range(6)]
            valeurs_ferme = valeurs + [valeurs[0]]
            angles_ferme = angles + [angles[0]]
            
            couleur = '#ff6b6b' if resultat == 1 else '#51cf66'
            
            ax.fill(angles_ferme, valeurs_ferme, alpha=0.3, color=couleur)
            ax.plot(angles_ferme, valeurs_ferme, linewidth=2, color=couleur)
            ax.scatter(angles, valeurs, color=couleur, s=50, zorder=5)
            
            for angle, val in zip(angles, valeurs):
                ax.annotate(f"{int(val*100)}%", 
                           xy=(angle, val), 
                           xytext=(angle, val + 0.12),
                           ha='center', va='center', 
                           fontsize=10, color='white', fontweight='bold')
            
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, size=10, color='white')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color='rgba(255,255,255,0.7)')
            ax.grid(color='rgba(255,255,255,0.2)', linestyle='--', linewidth=0.5)
            ax.spines['polar'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)

            # Barres
            st.markdown(f"### {T['detail']}")
            for nom, val in zip(T["cat_barres"], valeurs):
                pct = int(val * 100)
                emoji = "🔴" if pct >= 66 else ("🟡" if pct >= 33 else "🟢")
                st.markdown(f"**{nom}** {emoji} `{pct}%`")
                st.progress(val)

            # Explication IA
            st.markdown(f"### {T['pourquoi']}")
            if hasattr(model_to_use, 'feature_importances_'):
                feature_names = T["feature_names"]
                importances = model_to_use.feature_importances_
                indices = np.argsort(importances)[::-1]

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor('#f8f9fa')
                ax2.set_facecolor('#f8f9fa')
                bars = ax2.barh([feature_names[i] for i in indices[:5]],
                                [importances[i] for i in indices[:5]],
                                color='#667eea', edgecolor='white', height=0.6)
                for bar, val in zip(bars, [importances[i] for i in indices[:5]]):
                    ax2.text(val+0.003, bar.get_y()+bar.get_height()/2,
                             f"{val*100:.1f}%", va='center', fontsize=9)
                ax2.set_xlabel(T["importance_label"], fontsize=10)
                ax2.set_title(T["importance_titre"], fontsize=11, pad=12)
                ax2.set_xlim(0, max(importances) + 0.08)
                ax2.grid(axis='x', linestyle='--', alpha=0.3)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2)

                st.info(f"{T['analyse_info']} **{feature_names[indices[0]]}** "
                        f"({importances[indices[0]]*100:.1f}%) {T['et']} "
                        f"**{feature_names[indices[1]]}** ({importances[indices[1]]*100:.1f}%).")

            # PDF
            st.markdown(f"### {T['pdf_titre']}")
            importances_pdf = model_to_use.feature_importances_ if hasattr(model_to_use, 'feature_importances_') else np.ones(8)/8
            indices_pdf = np.argsort(importances_pdf)[::-1]
            pdf = generer_pdf(age, gender, family, work, remote, benefits, seek, anon,
                              score_risque, score_bienetre, resultat, valeurs,
                              importances_pdf, T["feature_names"], indices_pdf)
            st.download_button(label=T["pdf_btn"], data=pdf,
                               file_name=f"mindcheck_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf")

# ── Historique Supabase ───────────────────────────────────────────
st.markdown("---")
st.markdown(f"<h3 style='text-align:center;color:white'>{T['histo_titre']}</h3>", unsafe_allow_html=True)

try:
    response = supabase.table("historique")\
        .select("*")\
        .eq("user_id", st.session_state.user_id)\
        .order("created_at", desc=True)\
        .limit(10)\
        .execute()

    if response.data:
        df_hist = pd.DataFrame(response.data)
        df_hist = df_hist[["date","langue","age","genre","antecedents","impact_travail","score_risque","resultat"]]
        st.dataframe(df_hist, use_container_width=True)

        if len(response.data) >= 2:
            st.markdown(f"<h4 style='text-align:center;color:white'>{T['evolution']}</h4>", unsafe_allow_html=True)
            scores = [int(r["score_risque"].replace("%","")) for r in reversed(response.data)]
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            fig3.patch.set_facecolor('#764ba2')
            ax3.set_facecolor('#8b6db5')
            ax3.plot(range(len(scores)), scores, color='white', linewidth=2, marker='o', markersize=6)
            ax3.fill_between(range(len(scores)), scores, alpha=0.3, color='white')
            ax3.set_ylabel(T["score_label"], color='white', fontsize=10)
            ax3.set_xlabel(T["analyses_label"], color='white', fontsize=10)
            ax3.tick_params(colors='white')
            ax3.set_ylim(0, 100)
            ax3.grid(axis='y', linestyle='--', alpha=0.3, color='white')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_color('white')
            ax3.spines['bottom'].set_color('white')
            plt.tight_layout()
            st.pyplot(fig3)

        if st.button(T["effacer"]):
            supabase.table("historique")\
                .delete()\
                .eq("user_id", st.session_state.user_id)\
                .execute()
            st.success(T["effacer_ok"])
            st.rerun()
    else:
        st.info(T["histo_vide"])

except Exception as e:
    st.error(f"Erreur base de données : {e}")

# ── QR Code ───────────────────────────────────────────────────────
st.markdown("---")
add_install_qr()

# ── Footer ────────────────────────────────────────────────────────
st.markdown(f'<div class="footer">{T["footer"]}</div>', unsafe_allow_html=True)