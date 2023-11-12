package my.aplication.manejointeligentedepragas.Banco;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class Banco extends SQLiteOpenHelper{

        private static final String NOME_BANCO = "bancoLocal.db";
        private static final int VERSAO = 21;

        public static final String Usuario = "Usuario";
        public static final String Cod_Usuario = "Cod_Usuario";
        public static final String Email = "Email";
        public static final String Senha = "Senha";
        public static final String Nome = "Nome";
        public static final String Telefone = "Telefone";
        public static final String Tipo = "Tipo";

        public static final String Propriedade = "Propriedade";
        public static final String Cod_Propriedade = "Cod_Propriedade";
        public static final String NomePropriedade = "Nome";
        public static final String CidadePropriedade = "Cidade";
        public static final String EstadoPropriedade = "Estado";
        public static final String fk_Produtor_Cod_Produtor = "fk_Produtor_Cod_Produtor";

        public static final String Cultura = "Cultura";
        public static final String Cod_Cultura = "Cod_Cultura";
        public static final String TamanhoDaCultura = "TamanhoDaCultura";
        public static final String fk_Propriedade_Cod_Propriedade = "fk_Propriedade_Cod_Propriedade";
        public static final String fk_Planta_Cod_Planta = "fk_Planta_Cod_Planta";
        public static final String nomePlantaCultura = "nomePlanta";
        public static final String count_talhao = "count_talhao";

        public static final String Talhao = "Talhao";
        public static final String Cod_Talhao = "Cod_Talhao";
        public static final String nomeTalhao = "Nome";
        public static final String Aplicado = "Aplicado";
        public static final String fk_Cultura_Cod_Cultura = "fk_Cultura_Cod_Cultura";
        public static final String fk_Planta_Cod_Planta_Talhao = "fk_Planta_Cod_Planta";

        public static final String PresencaPraga = "PresencaPraga";
        public static final String Cod_PresencaPraga = "Cod_PresencaPraga";
        public static final String Status = "Status";
        public static final String fk_Praga_Cod_Praga = "fk_Praga_Cod_Praga";
        public static final String fk_Talhao_Cod_Talhao = "fk_Talhao_Cod_Talhao";
        public static final String Sync_Status = "Sync_Status";

        public static final String Atinge = "Atinge";
        public static final String Cod_Atinge = "Cod_Atinge";
        public static final String NivelDeControle = "NivelDeControle";
        public static final String NumeroPlantasAmostradas = "NumeroPlantasAmostradas";
        public static final String PontosPorTalhao = "PontosPorTalhao";
        public static final String PlantasPorPonto = "PlantasPorPonto";
        public static final String NumAmostras = "NumAmostras";

    //public static final String fk_Praga_Cod_Praga = "fk_Praga_Cod_Praga";
    //public static final String fk_Planta_Cod_Planta = "fk_Planta_Cod_Planta"

        public static final String Praga = "Praga";
        public static final String Cod_Praga = "Cod_Praga";
        public static final String NomePraga = "Nome";
        public static final String Familia = "Familia";
        public static final String Ordem = "Ordem";
        public static final String Descricao = "Descricao";
        public static final String NomeCientifico = "NomeCientifico";
        public static final String Localizacao = "Localizacao";
        public static final String AmbientePropicio = "AmbientePropicio";
        public static final String CicloVida = "CicloVida";
        public static final String Injurias = "Injurias";
        public static final String Observacoes = "Observacoes";
        public static final String HorarioDeAtuacao = "HorarioDeAtuacao";
        public static final String EstagioDeAtuacao = "EstagioDeAtuacao";
        public static final String ControleCultural = "ControleCultural";

        public static final String PlanoAmostragem = "PlanoAmostragem";
        public static final String Cod_Plano = "Cod_Plano";
        public static final String Autor = "Autor";
        public static final String Data = "Data";
        public static final String PlantasInfestadas = "PlantasInfestadas";
        public static final String PlantasAmostradas = "PlantasAmostradas";
        //public static final String fk_Talhao_Cod_Talhao = "fk_Talhao_Cod_Talhao";
        //public static final String fk_Praga_Cod_Praga = "fk_Praga_Cod_Praga";

        public static final String UltimosPlanos = "UltimosPlanos";
        //public static final String fk_Talhao_Cod_Talhao = "fk_Talhao_Cod_Talhao";
        //public static final String fk_Praga_Cod_Praga = "fk_Praga_Cod_Praga";

        public static final String Aplicacao = "Aplicacao";
        public static final String Cod_Aplicacao = "Cod_Aplicacao";
        //public static final String Autor = "Autor";
        //public static final String Data = "Data";
        public static final String fk_MetodoDeControle_Cod_MetodoDeControle = "fk_MetodoDeControle_Cod_MetodoDeControle";
        //public static final String fk_Talhao_Cod_Talhao = "fk_Talhao_Cod_Talhao";
        //public static final String fk_Praga_Cod_Praga = "fk_Praga_Cod_Praga";


        public Banco(Context context) {
            super(context, NOME_BANCO,null,VERSAO);
        }

        @Override
        public void onCreate(SQLiteDatabase db) {
            String sql = "CREATE TABLE "+Usuario+"("
                    + Cod_Usuario + " integer primary key,"
                    + Email+ " text,"
                    + Senha+ " text,"
                    + Nome+ " text,"
                    + Telefone+ " text,"
                    + Tipo+ " text"
                    + ")";

            String sql2 = "CREATE TABLE "+Propriedade+"("
                    + Cod_Propriedade + " integer primary key,"
                    + NomePropriedade+ " text,"
                    + CidadePropriedade+ " text,"
                    + EstadoPropriedade+ " text,"
                    + fk_Produtor_Cod_Produtor+ " integer"
                    + ")";

            String sql3 = "CREATE TABLE "+Cultura+"("
                    + Cod_Cultura + " integer primary key,"
                    + TamanhoDaCultura+ " float,"
                    + fk_Propriedade_Cod_Propriedade+ " integer,"
                    + fk_Planta_Cod_Planta+ " integer,"
                    + nomePlantaCultura+ " text,"
                    + count_talhao+ " integer"
                    + ")";

            String sql4 = "CREATE TABLE "+ Talhao +"("
                    + Cod_Talhao + " integer primary key,"
                    + nomeTalhao+ " text,"
                    + Aplicado+ " boolean,"
                    + fk_Cultura_Cod_Cultura+ " integer,"
                    + fk_Planta_Cod_Planta_Talhao+ " integer"
                    + ")";

            String sql5 = "CREATE TABLE "+ PresencaPraga +"("
                    + Cod_PresencaPraga + " integer primary key,"
                    + Status+ " integer,"
                    + fk_Praga_Cod_Praga+ " integer,"
                    + fk_Talhao_Cod_Talhao+ " integer,"
                    + Sync_Status+" integer"
                    + ")";

            String sql6 = "CREATE TABLE "+ Praga +"("
                    + Cod_Praga + " integer primary key,"
                    + NomePraga+ " text,"
                    + Familia+ " text,"
                    + Ordem+ " text,"
                    + Descricao+ " text,"
                    + NomeCientifico+ " text,"
                    + Localizacao+ " text,"
                    + AmbientePropicio+ " text,"
                    + CicloVida+ " text,"
                    + Injurias+ " text,"
                    + Observacoes+ " text,"
                    + HorarioDeAtuacao+ " text,"
                    + EstagioDeAtuacao+ " text,"
                    + ControleCultural+ " text"
                    + ")";

            String sql7 = "CREATE TABLE "+ Atinge +"("
                    + Cod_Atinge + " integer primary key,"
                    + NivelDeControle+ " float,"
                    + NumeroPlantasAmostradas+ " integer,"
                    + PontosPorTalhao+ " integer,"
                    + PlantasPorPonto+ " integer,"
                    + fk_Praga_Cod_Praga+ " integer,"
                    + fk_Planta_Cod_Planta+ " integer,"
                    + NumAmostras+ " integer"
                    + ")";

            String sql8 = "CREATE TABLE "+PlanoAmostragem+"("
                    + Cod_Plano + " integer primary key,"
                    + Autor+ " text,"
                    + Data+ " text,"
                    + PlantasInfestadas+ " integer,"
                    + PlantasAmostradas+ " integer,"
                    + fk_Talhao_Cod_Talhao+ " integer,"
                    + fk_Praga_Cod_Praga+ " integer,"
                    + Sync_Status+ " integer"
                    + ")";

            String sql9 = "CREATE TABLE "+UltimosPlanos+"("
                    + Data+ " text,"
                    + fk_Talhao_Cod_Talhao+ " integer,"
                    + fk_Praga_Cod_Praga+ " integer"
                    + ")";

            String sql0 = "CREATE TABLE "+Aplicacao+"("
                    + Cod_Aplicacao + " integer primary key,"
                    + Autor+ " text,"
                    + Data+ " text,"
                    + fk_MetodoDeControle_Cod_MetodoDeControle+ " integer,"
                    + fk_Talhao_Cod_Talhao+ " integer,"
                    + fk_Praga_Cod_Praga+ " integer"
                    + ")";

            db.execSQL(sql);
            db.execSQL(sql2);
            db.execSQL(sql3);
            db.execSQL(sql4);
            db.execSQL(sql5);
            db.execSQL(sql6);
            db.execSQL(sql7);
            db.execSQL(sql8);
            db.execSQL(sql9);
            db.execSQL(sql0);
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int i, int i1) {
            db.execSQL("DROP TABLE IF EXISTS "+ Usuario);
            db.execSQL("DROP TABLE IF EXISTS "+ Propriedade);
            db.execSQL("DROP TABLE IF EXISTS "+ Cultura);
            db.execSQL("DROP TABLE IF EXISTS "+ Talhao);
            db.execSQL("DROP TABLE IF EXISTS "+ PresencaPraga);
            db.execSQL("DROP TABLE IF EXISTS "+ Praga);
            db.execSQL("DROP TABLE IF EXISTS "+ Atinge);
            db.execSQL("DROP TABLE IF EXISTS "+ PlanoAmostragem);
            db.execSQL("DROP TABLE IF EXISTS "+ UltimosPlanos);
            db.execSQL("DROP TABLE IF EXISTS "+ Aplicacao);
            onCreate(db);
        }
    }