package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.PragaModel;

import java.util.ArrayList;

public class Controller_Praga {
    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Praga(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addPraga(PragaModel pragaModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Praga, pragaModel.getCod_Praga());
        cv.put(bancoLocal.NomePraga, pragaModel.getNome());
        cv.put(bancoLocal.Familia, pragaModel.getFamilia());
        cv.put(bancoLocal.Ordem, pragaModel.getOrdem());
        cv.put(bancoLocal.Descricao, pragaModel.getDescricao());
        cv.put(bancoLocal.NomeCientifico, pragaModel.getNomeCientifico());
        cv.put(bancoLocal.Localizacao, pragaModel.getLocalizacao());
        cv.put(bancoLocal.AmbientePropicio, pragaModel.getAmbientePropicio());
        cv.put(bancoLocal.CicloVida, pragaModel.getCicloVida());
        cv.put(bancoLocal.Injurias, pragaModel.getInjurias());
        cv.put(bancoLocal.Observacoes, pragaModel.getObservacoes());
        cv.put(bancoLocal.HorarioDeAtuacao, pragaModel.getHorarioDeAtuacao());
        cv.put(bancoLocal.EstagioDeAtuacao, pragaModel.getEstagioDeAtuacao());
        cv.put(bancoLocal.ControleCultural, pragaModel.getControleCultural());
        return db.insert(bancoLocal.Praga, null, cv) != -1;
    }

    public boolean updatePraga(PragaModel pragaModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Praga, pragaModel.getCod_Praga());
        cv.put(bancoLocal.NomePraga, pragaModel.getNome());
        cv.put(bancoLocal.Familia, pragaModel.getFamilia());
        cv.put(bancoLocal.Ordem, pragaModel.getOrdem());
        cv.put(bancoLocal.Descricao, pragaModel.getDescricao());
        cv.put(bancoLocal.NomeCientifico, pragaModel.getNomeCientifico());
        cv.put(bancoLocal.Localizacao, pragaModel.getLocalizacao());
        cv.put(bancoLocal.AmbientePropicio, pragaModel.getAmbientePropicio());
        cv.put(bancoLocal.CicloVida, pragaModel.getCicloVida());
        cv.put(bancoLocal.Injurias, pragaModel.getInjurias());
        cv.put(bancoLocal.Observacoes, pragaModel.getObservacoes());
        cv.put(bancoLocal.HorarioDeAtuacao, pragaModel.getHorarioDeAtuacao());
        cv.put(bancoLocal.EstagioDeAtuacao, pragaModel.getEstagioDeAtuacao());
        cv.put(bancoLocal.ControleCultural, pragaModel.getControleCultural());
        String where = " Cod_Praga = '"+ pragaModel.getCod_Praga()+"'";
        return db.update(bancoLocal.Praga, cv, where, null) > 0;
    }

    public ArrayList<PragaModel> getPraga(int codPraga) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Praga+" WHERE Cod_Praga=" +codPraga;
        ArrayList<PragaModel> pragas = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                PragaModel pragaModel = new PragaModel();
                pragaModel.setCod_Praga(c.getInt(0));
                pragaModel.setNome(c.getString(1));
                pragaModel.setFamilia(c.getString(2));
                pragaModel.setOrdem(c.getString(3));
                pragaModel.setDescricao(c.getString(4));
                pragaModel.setNomeCientifico(c.getString(5));
                pragaModel.setLocalizacao(c.getString(6));
                pragaModel.setAmbientePropicio(c.getString(7));
                pragaModel.setCicloVida(c.getString(8));
                pragaModel.setInjurias(c.getString(9));
                pragaModel.setObservacoes(c.getString(10));
                pragaModel.setHorarioDeAtuacao(c.getString(11));
                pragaModel.setEstagioDeAtuacao(c.getString(12));
                pragaModel.setControleCultural(c.getString(13));
                pragas.add(pragaModel);
            } while (c.moveToNext());
        }
        c.close();
        return pragas;
    }

    public void removerPraga(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Praga, query, null);
    }

    public void removerPragaEspecifica(PragaModel pragaModel){
        db = bancoLocal.getWritableDatabase();
        String where = " Cod_Praga = '"+ pragaModel.getCod_Praga()+"'";
        db.delete(bancoLocal.Praga, where, null);
    }

    public ArrayList<PragaModel> getPragaAtinge(int codPlanta) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Praga+" INNER JOIN "+bancoLocal.Atinge+" ON Praga.Cod_Praga=Atinge.fk_Praga_Cod_Praga WHERE Atinge.fk_Planta_Cod_Planta="+codPlanta+";";
        ArrayList<PragaModel> pragas = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                PragaModel pragaModel = new PragaModel();
                pragaModel.setCod_Praga(c.getInt(0));
                pragaModel.setNome(c.getString(1));
                pragaModel.setFamilia(c.getString(2));
                pragaModel.setOrdem(c.getString(3));
                pragaModel.setDescricao(c.getString(4));
                pragaModel.setNomeCientifico(c.getString(5));
                pragaModel.setLocalizacao(c.getString(6));
                pragaModel.setAmbientePropicio(c.getString(7));
                pragaModel.setCicloVida(c.getString(8));
                pragaModel.setInjurias(c.getString(9));
                pragaModel.setObservacoes(c.getString(10));
                pragaModel.setHorarioDeAtuacao(c.getString(11));
                pragaModel.setEstagioDeAtuacao(c.getString(12));
                pragaModel.setControleCultural(c.getString(13));
                pragas.add(pragaModel);
            } while (c.moveToNext());
        }
        c.close();
        return pragas;
    }

    public String getNome(int Cod_Praga) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Nome FROM " + bancoLocal.Praga +" WHERE Praga.Cod_Praga="+Cod_Praga+";";
        String nome = "";
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                nome = c.getString(0);
            } while (c.moveToNext());
        }
        c.close();
        return nome;
    }

    public String getAmostra(int Cod_Praga) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Localizacao FROM " + bancoLocal.Praga +" WHERE Praga.Cod_Praga="+Cod_Praga+";";
        String local = "";
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                local = c.getString(0);
            } while (c.moveToNext());
        }
        c.close();
        return local;
    }

    public ArrayList<String> getPragaAtingeNome(int codPlanta) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Nome FROM " + bancoLocal.Praga+" INNER JOIN "+bancoLocal.Atinge+" ON Praga.Cod_Praga=Atinge.fk_Praga_Cod_Praga WHERE Atinge.fk_Planta_Cod_Planta="+codPlanta+";";
        ArrayList<String> nomesPraga = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                nomesPraga.add(c.getString(0));
            } while (c.moveToNext());
        }
        c.close();
        return nomesPraga;
    }

    public ArrayList<Integer> getPragaAtingeCodigo(int codPlanta) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Cod_Praga FROM " + bancoLocal.Praga+" INNER JOIN "+bancoLocal.Atinge+" ON Praga.Cod_Praga=Atinge.fk_Praga_Cod_Praga WHERE Atinge.fk_Planta_Cod_Planta="+codPlanta+";";
        ArrayList<Integer> codPragas = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                codPragas.add(c.getInt(0));
            } while (c.moveToNext());
        }
        c.close();
        return codPragas;
    }

    public ArrayList<Integer> getPragaAtingeStatus(int codPlanta, int Cod_Talhao) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT PresencaPraga.Status FROM " + bancoLocal.PresencaPraga+" INNER JOIN "+bancoLocal.Praga+" ON PresencaPraga.fk_Praga_Cod_Praga=Atinge.fk_Praga_Cod_Praga WHERE Atinge.fk_Planta_Cod_Planta="+codPlanta+";";
        ArrayList<Integer> codPragas = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                codPragas.add(c.getInt(0));
            } while (c.moveToNext());
        }
        c.close();
        return codPragas;
    }

    public ArrayList<Integer> getStatusPresenca(int Cod_Talhao) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT PresencaPraga.Status FROM " + bancoLocal.PresencaPraga +" INNER JOIN "+bancoLocal.Praga+" ON PresencaPraga.fk_Praga_Cod_Praga=Praga.Cod_Praga WHERE PresencaPraga.fk_Talhao_Cod_Talhao="+Cod_Talhao+";";
        ArrayList<Integer> statusPraga = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                statusPraga.add(c.getInt(0));
            } while (c.moveToNext());
        }
        c.close();
        return statusPraga;
    }
    public ArrayList<Integer> getCodPresenca(int Cod_Talhao) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Cod_Praga FROM " + bancoLocal.Praga +" INNER JOIN "+bancoLocal.PresencaPraga+" ON Praga.Cod_Praga=PresencaPraga.fk_Praga_Cod_Praga WHERE PresencaPraga.fk_Talhao_Cod_Talhao="+Cod_Talhao+";";
        ArrayList<Integer> codPraga = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                codPraga.add(c.getInt(0));
            } while (c.moveToNext());
        }
        c.close();
        return codPraga;
    }
    public ArrayList<String> getNomePresenca(int Cod_Talhao) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Praga.Nome FROM " + bancoLocal.Praga +" INNER JOIN "+bancoLocal.PresencaPraga+" ON Praga.Cod_Praga=PresencaPraga.fk_Praga_Cod_Praga WHERE PresencaPraga.fk_Talhao_Cod_Talhao="+Cod_Talhao+";";
        ArrayList<String> nomePraga = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                nomePraga.add(c.getString(0));
            } while (c.moveToNext());
        }
        c.close();
        return nomePraga;
    }


}