package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.CulturaModel;

import java.util.ArrayList;

public class Controller_Cultura {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Cultura(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addCultura(CulturaModel culturaModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Cultura, culturaModel.getCod_Cultura());
        cv.put(bancoLocal.TamanhoDaCultura, culturaModel.getTamanhoCultura());
        cv.put(bancoLocal.fk_Planta_Cod_Planta, culturaModel.getFk_Cod_Planta());
        cv.put(bancoLocal.fk_Propriedade_Cod_Propriedade, culturaModel.getFk_Cod_Propriedade());
        cv.put(bancoLocal.nomePlantaCultura, culturaModel.getnomePlanta());
        cv.put(bancoLocal.count_talhao, culturaModel.getNumeroTalhoes());
        return db.insert(bancoLocal.Cultura, null, cv) != -1;
    }

    public boolean updateCultura(CulturaModel culturaModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Cultura, culturaModel.getCod_Cultura());
        cv.put(bancoLocal.TamanhoDaCultura, culturaModel.getTamanhoCultura());
        cv.put(bancoLocal.fk_Planta_Cod_Planta, culturaModel.getFk_Cod_Planta());
        cv.put(bancoLocal.fk_Propriedade_Cod_Propriedade, culturaModel.getFk_Cod_Propriedade());
        cv.put(bancoLocal.nomePlantaCultura, culturaModel.getnomePlanta());
        cv.put(bancoLocal.count_talhao, culturaModel.getNumeroTalhoes());
        String where = " Cod_Cultura = '"+ culturaModel.getCod_Cultura()+"'";
        return db.update(bancoLocal.Cultura, cv, where, null) > 0;
    }

    public ArrayList<CulturaModel> getCultura(int codPropriedade) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Cultura+" WHERE fk_Propriedade_Cod_Propriedade=" +codPropriedade+" ORDER BY "+bancoLocal.nomePlantaCultura;
        ArrayList<CulturaModel> culturas = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                CulturaModel culturaModel = new CulturaModel();
                culturaModel.setCod_Cultura(c.getInt(0));
                culturaModel.setTamanhoCultura(c.getFloat(1));
                culturaModel.setFk_Cod_Propriedade(c.getInt(2));
                culturaModel.setFk_Cod_Planta(c.getInt(3));
                culturaModel.setnomePlanta(c.getString(4));
                culturaModel.setNumeroTalhoes(c.getInt(5));
                culturas.add(culturaModel);
            } while (c.moveToNext());
        }
        c.close();
        return culturas;
    }

    public void removerCultura(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Cultura, query, null);
    }

    public void removerCulturaEspecifica(CulturaModel culturaModel){
        db = bancoLocal.getWritableDatabase();
        String where = " Cod_Cultura = '"+ culturaModel.getCod_Cultura()+"'";
        db.delete(bancoLocal.Cultura, where, null);
    }
}
