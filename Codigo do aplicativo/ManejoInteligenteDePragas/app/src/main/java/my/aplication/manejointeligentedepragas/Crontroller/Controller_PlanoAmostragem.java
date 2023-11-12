package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;

import java.util.ArrayList;

public class Controller_PlanoAmostragem {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_PlanoAmostragem(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addPlano(PlanoAmostragemModel planoAmostragemModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Plano, planoAmostragemModel.getCod_Plano());
        cv.put(bancoLocal.Autor, planoAmostragemModel.getAutor());
        cv.put(bancoLocal.Data, planoAmostragemModel.getDate());
        cv.put(bancoLocal.PlantasInfestadas, planoAmostragemModel.getPlantasInfestadas());
        cv.put(bancoLocal.PlantasAmostradas, planoAmostragemModel.getPlantasAmostradas());
        cv.put(bancoLocal.fk_Talhao_Cod_Talhao, planoAmostragemModel.getFk_Cod_Talhao());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, planoAmostragemModel.getFk_Cod_Praga());
        cv.put(bancoLocal.Sync_Status, planoAmostragemModel.getSync_Status());
        return db.insert(bancoLocal.PlanoAmostragem, null, cv) != -1;
    }

    public boolean updatePlano(PlanoAmostragemModel planoAmostragemModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Plano, planoAmostragemModel.getCod_Plano());
        cv.put(bancoLocal.Autor, planoAmostragemModel.getAutor());
        cv.put(bancoLocal.Data, planoAmostragemModel.getDate());
        cv.put(bancoLocal.PlantasInfestadas, planoAmostragemModel.getPlantasInfestadas());
        cv.put(bancoLocal.PlantasAmostradas, planoAmostragemModel.getPlantasAmostradas());
        cv.put(bancoLocal.fk_Talhao_Cod_Talhao, planoAmostragemModel.getFk_Cod_Talhao());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, planoAmostragemModel.getFk_Cod_Praga());
        cv.put(bancoLocal.Sync_Status, planoAmostragemModel.getSync_Status());
        String where = " Cod_Plano = '"+ planoAmostragemModel.getCod_Plano()+"'";
        return db.update(bancoLocal.PlanoAmostragem, cv, where, null) > 0;
    }

    public ArrayList<PlanoAmostragemModel> getPlano() {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.PlanoAmostragem;
        ArrayList<PlanoAmostragemModel> planos = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                PlanoAmostragemModel planoAmostragemModel = new PlanoAmostragemModel();
                planoAmostragemModel.setCod_Plano(c.getInt(0));
                planoAmostragemModel.setAutor(c.getString(1));
                planoAmostragemModel.setDate(c.getString(2));
                planoAmostragemModel.setPlantasInfestadas(c.getInt(3));
                planoAmostragemModel.setPlantasAmostradas(c.getInt(4));
                planoAmostragemModel.setFk_Cod_Talhao(c.getInt(5));
                planoAmostragemModel.setFk_Cod_Praga(c.getInt(6));
                planoAmostragemModel.setSync_Status(c.getInt(7));
                planos.add(planoAmostragemModel);
            } while (c.moveToNext());
        }
        c.close();
        return planos;
    }

    public ArrayList<PlanoAmostragemModel> getPlanoOffline() {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.PlanoAmostragem + " WHERE PlanoAmostragem.Sync_Status = 1"+";";
        ArrayList<PlanoAmostragemModel> planos = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                PlanoAmostragemModel planoAmostragemModel = new PlanoAmostragemModel();
                planoAmostragemModel.setCod_Plano(c.getInt(0));
                planoAmostragemModel.setAutor(c.getString(1));
                planoAmostragemModel.setDate(c.getString(2));
                planoAmostragemModel.setPlantasInfestadas(c.getInt(3));
                planoAmostragemModel.setPlantasAmostradas(c.getInt(4));
                planoAmostragemModel.setFk_Cod_Talhao(c.getInt(5));
                planoAmostragemModel.setFk_Cod_Praga(c.getInt(6));
                planoAmostragemModel.setSync_Status(c.getInt(7));
                planos.add(planoAmostragemModel);
            } while (c.moveToNext());
        }
        c.close();
        return planos;
    }

    public void removerPlano(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.PlanoAmostragem, query, null);
    }
}
