package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.PropriedadeModel;

import java.util.ArrayList;

public class Controller_Propriedade {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Propriedade(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addPropriedade(PropriedadeModel propriedadeModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Propriedade, propriedadeModel.getCod_Propriedade());
        cv.put(bancoLocal.NomePropriedade, propriedadeModel.getNome());
        cv.put(bancoLocal.CidadePropriedade, propriedadeModel.getCidade());
        cv.put(bancoLocal.EstadoPropriedade, propriedadeModel.getEstado());
        cv.put(bancoLocal.fk_Produtor_Cod_Produtor, propriedadeModel.getFk_Cod_Produtor());
        return db.insert(bancoLocal.Propriedade, null, cv) != -1;
    }

    public boolean updatePropriedade(PropriedadeModel propriedadeModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Propriedade, propriedadeModel.getCod_Propriedade());
        cv.put(bancoLocal.NomePropriedade, propriedadeModel.getNome());
        cv.put(bancoLocal.CidadePropriedade, propriedadeModel.getCidade());
        cv.put(bancoLocal.EstadoPropriedade, propriedadeModel.getEstado());
        cv.put(bancoLocal.fk_Produtor_Cod_Produtor, propriedadeModel.getFk_Cod_Produtor());
        String where = " Cod_Propriedade = '"+ propriedadeModel.getCod_Propriedade()+"'";
        return db.update(bancoLocal.Propriedade, cv, where, null) > 0;
    }

    public ArrayList<PropriedadeModel> getPropriedade() {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Propriedade;
        ArrayList<PropriedadeModel> propriedades = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                PropriedadeModel Propriedade = new PropriedadeModel();
                Propriedade.setCod_Propriedade(c.getInt(0));
                Propriedade.setNome(c.getString(1));
                Propriedade.setCidade(c.getString(2));
                Propriedade.setEstado(c.getString(3));
                Propriedade.setFk_Cod_Produtor(c.getInt(4));
                propriedades.add(Propriedade);
            } while (c.moveToNext());
        }
        c.close();
        return propriedades;
    }

    public void removerPropriedade(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Propriedade, query, null);
    }
}
