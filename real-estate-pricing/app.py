import os
import pandas as pd
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from enum import Enum
from fastapi import FastAPI, Request, Header, HTTPException, Body
from fastapi.responses import RedirectResponse
from typing import Iterable, Tuple, Optional
from itertools import chain

project_path = Path.cwd()
metadata = bootstrap_project(project_path)
session = KedroSession.create(metadata.package_name, project_path)
context = session.load_context()

app = FastAPI(
    title="FastAPI",
    version="0.1.0",
    description="""
        ChimichangApp API helps you do awesome stuff. 🚀
## Items
You can **read items**.
## Users
You will be able to:
* **Create users** (_not implemented_). * **Read users** (_not implemented_).

    """,
    openapi_tags=[],
)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


MLPredictor = context.catalog.load("MLPredictor")


class MLPredictorshp_bairro_distrito(str, Enum):
    Itaim_Bibi = "Itaim_Bibi"
    Pinheiros = "Pinheiros"
    Jd_Paulista = "Jd_Paulista"
    Lapa = "Lapa"
    Perdizes = "Perdizes"
    Santa_Cecília = "Santa_Cecília"
    Ipiranga = "Ipiranga"
    Morumbi = "Morumbi"
    Vl_Leopoldina = "Vl_Leopoldina"
    Vl_Mariana = "Vl_Mariana"
    Barra_Funda = "Barra_Funda"
    Bela_Vista = "Bela_Vista"
    Moema = "Moema"
    Cursino = "Cursino"
    Brás = "Brás"
    Rio_Pequeno = "Rio_Pequeno"
    Santo_Amaro = "Santo_Amaro"
    Penha = "Penha"
    Consolação = "Consolação"
    Jabaquara = "Jabaquara"
    Liberdade = "Liberdade"
    Mandaqui = "Mandaqui"
    Vl_Formosa = "Vl_Formosa"
    Vl_Andrade = "Vl_Andrade"
    Campo_Belo = "Campo_Belo"
    Vl_Matilde = "Vl_Matilde"
    Sacomã = "Sacomã"
    Limão = "Limão"
    Aricanduva = "Aricanduva"
    Vl_Sonia = "Vl_Sonia"
    Cambuci = "Cambuci"
    Vl_Maria = "Vl_Maria"
    Alto_de_Pinheiros = "Alto_de_Pinheiros"
    Saúde = "Saúde"
    Jaguaré = "Jaguaré"
    Tatuapé = "Tatuapé"
    Cangaíba = "Cangaíba"
    Carrão = "Carrão"
    Tremembé = "Tremembé"
    Cidade_Ademar = "Cidade_Ademar"
    Santana = "Santana"
    Cidade_Tiradentes = "Cidade_Tiradentes"
    Mooca = "Mooca"
    Vl_Medeiros = "Vl_Medeiros"
    Freguesia_do_Ó = "Freguesia_do_Ó"
    República = "República"
    Vl_Guilherme = "Vl_Guilherme"
    Casa_Verde = "Casa_Verde"
    Pedreira = "Pedreira"
    Belém = "Belém"
    Butantã = "Butantã"
    Jd_São_Luis = "Jd_São_Luis"
    Vl_Prudente = "Vl_Prudente"
    Campo_Grande = "Campo_Grande"
    Cidade_Dutra = "Cidade_Dutra"
    Tucuruvi = "Tucuruvi"
    Água_Rasa = "Água_Rasa"
    Socorro = "Socorro"
    Jáguara = "Jáguara"
    Cachoeirinha = "Cachoeirinha"
    Itaquera = "Itaquera"
    Cidade_Líder = "Cidade_Líder"
    Artur_Alvim = "Artur_Alvim"
    Raposo_Tavares = "Raposo_Tavares"
    Pirituba = "Pirituba"
    Ermelino_Matarazzo = "Ermelino_Matarazzo"
    São_Lucas = "São_Lucas"
    Grajaú = "Grajaú"
    Jaçana = "Jaçana"
    Campo_Limpo = "Campo_Limpo"
    Parelheiros = "Parelheiros"
    Bom_Retiro = "Bom_Retiro"
    Ponte_Rasa = "Ponte_Rasa"
    Guaianases = "Guaianases"
    Sapopemba = "Sapopemba"
    São_Domingos = "São_Domingos"
    Parque_do_Carmo = "Parque_do_Carmo"
    Sé = "Sé"
    São_Miguel = "São_Miguel"
    Jd_Ângela = "Jd_Ângela"
    Vl_Jacui = "Vl_Jacui"
    Jaraguá = "Jaraguá"
    São_Mateus = "São_Mateus"
    Capão_Redondo = "Capão_Redondo"
    Brasilândia = "Brasilândia"
    Vl_Curuca = "Vl_Curuca"
    Pari = "Pari"
    São_Rafael = "São_Rafael"
    Anhanguera = "Anhanguera"
    Jd_Helena = "Jd_Helena"
    Perus = "Perus"
    Lajeado = "Lajeado"
    Iguatemi = "Iguatemi"
    Itaim_Paulista = "Itaim_Paulista"
    José_Bonifacio = "José_Bonifacio"


class MLPredictortipo_imovel(str, Enum):
    APARTAMENTO = "APARTAMENTO"
    CASA = "CASA"


@app.get("/sales_and_rental_prediction", tags=["users"])
def predict_sales_and_rental_prediction(
    area_util: float,
    dormitorios: int,
    suites: int,
    banheiros: int,
    vagas: int,
    salas: int,
    shp_bairro_distrito: MLPredictorshp_bairro_distrito,
    tipo_imovel: MLPredictortipo_imovel,
    ano_construcao: int,
):
    args = {
        "area_util": area_util,
        "dormitorios": dormitorios,
        "suites": suites,
        "banheiros": banheiros,
        "vagas": vagas,
        "salas": salas,
        "shp_bairro_distrito": shp_bairro_distrito,
        "tipo_imovel": tipo_imovel,
        "ano_construcao": ano_construcao,
    }
    df = pd.DataFrame({k: [v] for k, v in args.items()})
    result = MLPredictor.predict(df, context)

    if result.get("error"):
        raise HTTPException(
            status_code=int(result.get("error").get("status_code")),
            detail=result.get("error").get("detail"),
        )

    return result


def _get_values_as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(chain.from_iterable(value.split(",") for value in values))


@app.post("/kedro")
def kedro(
    request: dict = Body(
        ...,
        example={
            "pipeline_name": "",
            "tag": [],
            "node_names": [],
            "from_nodes": [],
            "to_nodes": [],
            "from_inputs": [],
            "to_outputs": [],
            "params": {},
        },
    )
):
    pipeline_name = request.get("pipeline_name")
    tag = request.get("tag")
    node_names = request.get("node_names")
    from_nodes = request.get("from_nodes")
    to_nodes = request.get("to_nodes")
    from_inputs = request.get("from_inputs")
    to_outputs = request.get("to_outputs")
    params = request.get("params")

    tag = _get_values_as_tuple(tag) if tag else tag
    node_names = _get_values_as_tuple(node_names) if node_names else node_names
    package_name = str(Path(__file__).resolve().parent.name)
    try:
        with KedroSession.create(
            package_name, env=None, extra_params=params
        ) as session:
            return session.run(
                tags=tag,
                node_names=node_names,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                from_inputs=from_inputs,
                to_outputs=to_outputs,
                pipeline_name=pipeline_name,
            )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/catalog")
def catalog(name: str):
    try:
        file = context.catalog.load(name)
        return file.to_json(force_ascii=False)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
