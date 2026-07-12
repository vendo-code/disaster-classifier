"""
Data deskripsi 6 kelas bencana untuk halaman Info Kelas.
"""

CLASS_INFO = {
    "Damaged_Infrastructure": {
        "label_id": "Infrastruktur Rusak",
        "emoji": "🏚️",
        "deskripsi": (
            "Kategori ini mencakup kerusakan pada bangunan, jembatan, jalan, "
            "dan infrastruktur buatan manusia lainnya akibat bencana alam atau kejadian ekstrem. "
            "Ciri visual yang umum meliputi dinding retak, atap roboh, puing-puing bangunan, "
            "dan struktur yang miring atau ambruk."
        ),
        "ciri_gradcam": (
            "Model cenderung fokus pada tepi bangunan yang tidak beraturan, "
            "retakan pada dinding, dan area puing atau reruntuhan."
        ),
        "warna": "#e67e22",
        "contoh_kejadian": "Gempa bumi, tsunami, ledakan, banjir bandang",
        "catatan_model": (
            "Kelas ini memiliki tingkat kebingungan dengan Land_Disaster "
            "karena sama-sama menampilkan lingkungan yang hancur."
        ),
    },
    "Fire_Disaster": {
        "label_id": "Bencana Kebakaran",
        "emoji": "🔥",
        "deskripsi": (
            "Bencana kebakaran meliputi kebakaran hutan, kebakaran gedung, dan kebakaran besar "
            "lainnya yang mengancam nyawa dan properti. "
            "Ciri visual yang khas adalah adanya api, asap tebal, dan area yang terbakar atau menghitam."
        ),
        "ciri_gradcam": (
            "Model mengaktifkan area yang mengandung warna merah-oranye api, "
            "asap hitam, dan area gelap bekas kebakaran."
        ),
        "warna": "#e74c3c",
        "contoh_kejadian": "Kebakaran hutan, kebakaran pemukiman, kebakaran industri",
        "catatan_model": "Salah satu kelas dengan performa terbaik berkat ciri visual yang sangat distinktif.",
    },
    "Human_Damage": {
        "label_id": "Korban Manusia",
        "emoji": "🚑",
        "deskripsi": (
            "Kategori ini menangkap situasi di mana dampak bencana langsung terlihat pada manusia, "
            "seperti korban yang membutuhkan pertolongan, kerumunan pengungsi, atau situasi evakuasi darurat. "
            "Berfokus pada keberadaan manusia dalam konteks bencana."
        ),
        "ciri_gradcam": (
            "Model memfokuskan perhatian pada sosok manusia, kerumunan, "
            "dan interaksi manusia dengan lingkungan bencana."
        ),
        "warna": "#9b59b6",
        "contoh_kejadian": "Penyelamatan korban banjir, evakuasi gempa, korban longsor",
        "catatan_model": "Performa baik namun bergantung pada ketampakan sosok manusia dalam citra.",
    },
    "Land_Disaster": {
        "label_id": "Bencana Tanah",
        "emoji": "⛰️",
        "deskripsi": (
            "Bencana tanah mencakup tanah longsor, erosi tebing, pergerakan tanah masif, "
            "dan likuefaksi. Ciri visual meliputi lereng yang ambruk, material tanah yang bergeser, "
            "dan area yang tertimbun."
        ),
        "ciri_gradcam": (
            "Model mengaktifkan area tanah/batuan yang bergeser, tepi tebing yang tidak stabil, "
            "dan permukaan tanah yang terganggu."
        ),
        "warna": "#795548",
        "contoh_kejadian": "Longsor, likuefaksi, erosi tebing akibat hujan deras",
        "catatan_model": (
            "Kelas ini memiliki F1-Score terendah (0,6705) dalam evaluasi model. "
            "Sering tertukar dengan Damaged_Infrastructure karena keduanya menampilkan "
            "lingkungan yang hancur/terganggu. Gunakan konteks tambahan saat menginterpretasi hasil ini."
        ),
    },
    "Non_Damage": {
        "label_id": "Tidak Ada Kerusakan",
        "emoji": "✅",
        "deskripsi": (
            "Kategori ini mewakili gambar yang tidak menunjukkan tanda-tanda bencana atau kerusakan. "
            "Lingkungan tampak normal, aman, dan tidak ada indikasi kejadian bencana. "
            "Berguna sebagai baseline untuk membedakan area aman dari area terdampak."
        ),
        "ciri_gradcam": (
            "Aktivasi Grad-CAM cenderung tersebar merata atau pada detail tekstur normal "
            "tanpa fokus spesifik pada area kerusakan."
        ),
        "warna": "#2ecc71",
        "contoh_kejadian": "Lingkungan perkotaan normal, pemandangan alam tidak terdampak",
        "catatan_model": "Kelas dengan presisi tinggi; model jarang salah mengklasifikasi bencana sebagai Non_Damage.",
    },
    "Water_Disaster": {
        "label_id": "Bencana Air",
        "emoji": "🌊",
        "deskripsi": (
            "Bencana air meliputi banjir, tsunami, banjir bandang, dan luapan air yang menyebabkan kerusakan. "
            "Ciri visual utama adalah genangan air yang luas, benda-benda terendam, "
            "dan perubahan dramatis pada badan air."
        ),
        "ciri_gradcam": (
            "Model mengaktifkan area yang menampilkan air atau genangan besar, "
            "objek yang terendam air, dan refleksi permukaan air."
        ),
        "warna": "#3498db",
        "contoh_kejadian": "Banjir perkotaan, tsunami, banjir bandang, luapan sungai",
        "catatan_model": "Performa sangat baik; ciri visual air yang menonjol membantu model membedakan kelas ini.",
    },
}
