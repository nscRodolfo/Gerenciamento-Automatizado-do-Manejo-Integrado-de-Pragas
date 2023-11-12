package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;
import android.os.Handler;
//import android.support.design.widget.FloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;
import android.os.Bundle;
//import android.support.v7.widget.LinearLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Propriedade;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.RecyclerViewAdapter.PropriedadeCardAdapter;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;
import my.aplication.manejointeligentedepragas.model.PropriedadeModel;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;


public class Propriedades extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    public FloatingActionButton fab;
    public TextView textView;
    public String nomePropriedade;
    public String tipoUsu;
    private Dialog mDialog;

    private DrawerLayout drawerLayout;

    //vars relative layout
    private ArrayList<PropriedadeModel> cards = new ArrayList<>();

    private ArrayList<PropriedadeModel> web = new ArrayList<>();
    private ArrayList<PropriedadeModel> local = new ArrayList<>();

    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();
    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();


    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_propriedades);
        openDialog();

        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        Controller_Usuario cu = new Controller_Usuario(getBaseContext());
        tipoUsu = cu.getUser().getTipo();

        fab = findViewById(R.id.fabAddProp);
        textView = findViewById(R.id.tvAddPropriedade);

        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            if(tipoUsu.equals("Funcionario")){
                setTitle("MIP² | Propriedades vinculadas");
                fab.hide();
                textView.setVisibility(View.GONE);
            }else if(tipoUsu.equals("Produtor")){
                setTitle("MIP² | Propriedades");
            }else if(tipoUsu.equals("Adm")){

            }
            mDialog.dismiss();
            Controller_Propriedade cp = new Controller_Propriedade(Propriedades.this);
            cards = cp.getPropriedade();
            fab.hide();
            textView.setVisibility(View.GONE);
            iniciarRecyclerView();
        }else {
            if(tipoUsu.equals("Funcionario")){
                setTitle("MIP² | Propriedades vinculadas");
                fab.hide();
                textView.setVisibility(View.GONE);
                resgatarDadosFunc(cu.getUser().getCod_Usuario());
            }else if(tipoUsu.equals("Produtor")){
                setTitle("MIP² | Propriedades");
                resgatarDados(cu.getUser().getCod_Usuario());
            }else if(tipoUsu.equals("Adm")){

            }

        }

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_propriedades);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_propriedades);
        NavigationView navigationView = findViewById(R.id.nav_view_propriedades);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);






        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Propriedades.this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }else {
                    Intent i = new Intent(Propriedades.this, AdicionarPropriedade.class);
                    startActivity(i);
                }
            }
        });


        textView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Propriedades.this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }else {
                    Intent i = new Intent(Propriedades.this, AdicionarPropriedade.class);
                    startActivity(i);
                }
            }
        });
    }


    // mensagem para sair do aplicativo, não volta mais no login
    private boolean backPressedOnce = false;
    private Handler backPressedHandler = new Handler();

    private static final int BACK_PRESSED_DELAY = 2000;

    private final Runnable backPressedTimeoutAction = new Runnable() {
        @Override
        public void run() {
            backPressedOnce = false;
        }
    };

    @Override
    protected void onStart() {
        super.onStart();
        ConnectionBuddy.getInstance().registerForConnectivityEvents(this, new ConnectivityChangeListener() {
            @Override
            public void onConnectionChange(ConnectivityEvent event) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    //Toast.makeText(Propriedades.this,"Você está offline!", Toast.LENGTH_LONG).show();
                }else{
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(Propriedades.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(Propriedades.this);

                    //Toast.makeText(Propriedades.this,"Você está online!", Toast.LENGTH_LONG).show();

                    planoAmostragemModels = cpa.getPlanoOffline();
                    presencaPragaModels = cpp.getPresencaPragaOffline();

                    for(int i=0; i<planoAmostragemModels.size(); i++){
                        SalvarPlanos(planoAmostragemModels.get(i));
                    }
                    cpa.removerPlano();

                    for(int i=0; i<presencaPragaModels.size(); i++){
                        SalvarPresencas(presencaPragaModels.get(i));
                    }
                    cpp.updatePresencaSyncStatus();

                }
            }
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
        ConnectionBuddy.getInstance().unregisterFromConnectivityEvents(this);
    }


    public void SalvarPlanos(PlanoAmostragemModel pam){
        Controller_Usuario cu = new Controller_Usuario(Propriedades.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(Propriedades.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Propriedades.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(Propriedades.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Propriedades.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            if (this.backPressedOnce) {
                // Finaliza a aplicacao
                finishAffinity();
                //System.exit(0);
                return;
            }
            this.backPressedOnce = true;
            Toast.makeText(this, "Pressione novamente para fechar",Toast.LENGTH_SHORT).show();
            backPressedHandler.postDelayed(backPressedTimeoutAction, BACK_PRESSED_DELAY);
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                Toast.makeText(Propriedades.this,"Você já está na tela de propriedades.", Toast.LENGTH_LONG).show();
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }



    private void resgatarDados(int codUsuario){

        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
            mDialog.dismiss();
        }else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarPropriedades.php?Cod_Usuario=" + codUsuario;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        try {
                            //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                            JSONArray array = new JSONArray(response);
                            for (int i = 0; i< array.length(); i++){
                                JSONObject obj = array.getJSONObject(i);
                                PropriedadeModel u = new PropriedadeModel();
                                u.setCod_Propriedade(obj.getInt("Cod_Propriedade"));
                                u.setNome(obj.getString("Nome"));
                                u.setCidade(obj.getString("Cidade"));
                                u.setEstado(obj.getString("Estado"));
                                u.setFk_Cod_Produtor(obj.getInt("fk_Produtor_Cod_Produtor"));
                                web.add(u);
                            }
                            Controller_Propriedade cp= new Controller_Propriedade(Propriedades.this);
                            local = cp.getPropriedade();
                             if(web.size() == local.size()) {
                                 if(!web.equals(local)) {
                                     cp.removerPropriedade();
                                     local = web;
                                     for (int i = 0; i < local.size(); i++) {
                                         cp.addPropriedade(local.get(i));
                                     }
                                     cards = local;
                                 }else{
                                     cards = local;
                                 }
                             }else{
                                 cp.removerPropriedade();
                                 local = web;
                                 for (int i = 0; i < local.size(); i++) {
                                     cp.addPropriedade(local.get(i));
                                 }
                                 cards = local;
                             }
                             iniciarRecyclerView();
                             mDialog.dismiss();
                        } catch (JSONException e) {
                            Toast.makeText(Propriedades.this, e.toString(), Toast.LENGTH_LONG).show();
                            mDialog.dismiss();
                        }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Propriedades.this,error.toString(), Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }
            }));

        }
        /*
        PropriedadeModel p1 = new PropriedadeModel(0,"Espera Feliz","Rio Pomba","MG",0);
        */
    }





    private void iniciarRecyclerView(){
        RecyclerView rv = findViewById(R.id.RVPropriedade);
        PropriedadeCardAdapter adapter = new PropriedadeCardAdapter(this, cards);
        rv.setAdapter(adapter);
        rv.setLayoutManager(new LinearLayoutManager(this));

    }


    private void resgatarDadosFunc(int codUsuario){

        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
            mDialog.dismiss();
        }else { // se tem acesso à internet
            String url = "https://mip.software/resgatarPropriedadesFunc.php?Cod_Usuario=" + codUsuario;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            PropriedadeModel u = new PropriedadeModel();
                            u.setCod_Propriedade(obj.getInt("Cod_Propriedade"));
                            u.setNome(obj.getString("Nome"));
                            u.setCidade(obj.getString("Cidade"));
                            u.setEstado(obj.getString("Estado"));
                            u.setFk_Cod_Produtor(obj.getInt("fk_Produtor_Cod_Produtor"));
                            web.add(u);
                        }
                        Controller_Propriedade cp= new Controller_Propriedade(Propriedades.this);
                        local = cp.getPropriedade();
                        if(web.size() == local.size()) {
                            if(!web.equals(local)) {
                                cp.removerPropriedade();
                                local = web;
                                for (int i = 0; i < local.size(); i++) {
                                    cp.addPropriedade(local.get(i));
                                }
                                cards = local;
                            }else{
                                cards = local;
                            }
                        }else{
                            cp.removerPropriedade();
                            local = web;
                            for (int i = 0; i < local.size(); i++) {
                                cp.addPropriedade(local.get(i));
                            }
                            cards = local;
                        }
                        if(cards.isEmpty()){
                            AlertDialog.Builder dlgBox = new AlertDialog.Builder(Propriedades.this);
                            dlgBox.setTitle("Aviso!");
                            dlgBox.setMessage("Você não está vinculado à nenhuma propriedade.");
                            dlgBox.setPositiveButton("Entendi", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    // não faz nada
                                }
                            });
                            dlgBox.show();
                        }else{
                            iniciarRecyclerView();
                        }
                        mDialog.dismiss();
                    } catch (JSONException e) {
                        Toast.makeText(Propriedades.this, e.toString(), Toast.LENGTH_LONG).show();
                        mDialog.dismiss();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Propriedades.this,error.toString(), Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }
            }));

        }
    }


    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }

}
