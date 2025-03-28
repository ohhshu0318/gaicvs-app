from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column, ForeignKey
from sqlalchemy.types import TIMESTAMP, String

Base = declarative_base()


class Case(Base):
    __tablename__ = "tblkj01"
    __table_args__ = {"schema": "kujo"}
    voice_no = Column("kjoinfo_tsuban", String, primary_key=True)
    classification_code = Column(
        "mshd_daibunric", String, ForeignKey("tblimdnai.mshd_daibunric")
    )
    category_code = Column(
        "mosd_chbnrui_c", String, ForeignKey("tblimdnai.mosd_chbnrui_c")
    )
    subcategory_code = Column(
        "mosd_shbnrui_c", String, ForeignKey("tblimdnai.mosd_shbnrui_c")
    )
    subsubcategory_code = Column(
        "mosd_saibnruic", String, ForeignKey("tblimdnai.mosd_saibnruic")
    )


class LabelMaster(Base):
    __tablename__ = "tblimdnai"
    __table_args__ = {"schema": "kujo"}
    classification_code = Column("mshd_daibunric", String, primary_key=True)
    category_code = Column("mosd_chbnrui_c", String, primary_key=True)
    subcategory_code = Column("mosd_shbnrui_c", String, primary_key=True)
    subsubcategory_code = Column("mosd_saibnruic", String, primary_key=True)
    classification_name = Column("msd_daibnrmei", String)
    category_name = Column("mosdchbnrui", String)
    subcategory_name = Column("mosd_shbnrui", String)
    subsubcategory_name = Column("msdsaibnrmeikn", String)


class GenaiClassificationResults(Base):
    __tablename__ = "tblgenbnrui"
    __table_args__ = {"schema": "kujo"}
    voice_no = Column("kjoinfo_tsuban", String, primary_key=True)
    classification_code = Column("genai_mshd_daibunric", String)
    category_code = Column("genai_mosd_chbnrui_c", String)
    subcategory_code = Column("genai_mosd_shbnrui_c", String)
    subsubcategory_code = Column("genai_mosd_saibnruic", String)
    classification_name = Column("genai_msd_daibnrmei", String)
    category_name = Column("genai_mosdchbnrui", String)
    subcategory_name = Column("genai_mosd_shbnrui", String)
    subsubcategory_name = Column("genai_msdsaibnrmei", String)
    created_datetime = Column("tourokutstnp", TIMESTAMP)
    created_by_division_code = Column("trksha_shzkc", String)
    created_by_division_name = Column("trksha_shzkmei", String)
    created_by_code = Column("tourokushacode", String)
    created_by_name = Column("tourokushamei", String)
    last_modified_datetime = Column("finalupd_tstnp", TIMESTAMP)
    last_modified_by_division_code = Column("finalupopshzkc", String)
    last_modified_by_division_name = Column("finluposhzkmei", String)
    last_modified_by_code = Column("final_upd_op_c", String)
    last_modified_by_name = Column("saishuksnsmei", String)
    deleted_flag = Column("sakujo_flg", String)
    deleted_datetime = Column("sakujo_tstnp", TIMESTAMP)
    deleted_by_division_code = Column("del_op_shzk_c", String)
    deleted_by_division_name = Column("del_op_shzkmei", String)
    deleted_by_code = Column("sakujosha_code", String)
    deleted_by_name = Column("sakujoshamei", String)
